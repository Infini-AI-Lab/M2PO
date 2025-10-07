import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import sys

_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}

def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/code_contests')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    dataset = datasets.load_dataset("deepmind/code_contests")

    train_dataset = dataset['train'].shuffle(seed=args.seed)
    if args.num_samples:
        train_dataset = train_dataset.select(range(args.num_samples))
    else:
        train_dataset = train_dataset.select(range(int(args.subset_ratio * len(train_dataset))))

    test_dataset = dataset["valid"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if "<image>" in example["description"]:
                print("Description includes image, skipping...")
                return _EMPTY_RETURN_

            stdin_list = (example["public_tests"]["input"] + example["private_tests"]["input"] +
                          example["generated_tests"]["input"])
            stdout_list = (example["public_tests"]["output"] + example["private_tests"]["output"] +
                           example["generated_tests"]["output"])

            # stdin_list, stdout_list = minimize_stdio(stdin_list, stdout_list, max_n_tests)
            assert len(stdin_list) == len(stdout_list)
            if len(stdin_list) == 0:
                return _EMPTY_RETURN_

            prompt = ("Solve the programming task below in a Python markdown code block. "
                      "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
                      "produces outputs through STDOUT (like those in the 'Output' section)."
                      f"\n\n{example['description'].strip()}")
            return {
                "data_source": "codecontests",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({
                        "inputs": stdin_list,
                        "outputs": stdout_list
                    }),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (example["solutions"]["solution"][0] if example["solutions"]["solution"] else ""),
                    "dataset": "deepmind/code_contests",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"),
                                      with_indices=True,
                                      remove_columns=train_dataset.column_names).filter(lambda x: x != _EMPTY_RETURN_)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset = train_dataset.filter(lambda x: x["prompt"] is not None)
    test_dataset = test_dataset.filter(lambda x: x["prompt"] is not None)

    # print length of train_dataset and test_dataset
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.num_samples:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.num_samples}_samples.parquet'))
    else:
        if args.subset_ratio == 1.0:
            train_dataset.to_parquet(os.path.join(local_dir, f'train_full.parquet'))
        else:
            train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.subset_ratio}.parquet'))

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # print data source and length
    print(f"Data source: codecontests")
    print(f"Length of train dataset: {len(train_dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
