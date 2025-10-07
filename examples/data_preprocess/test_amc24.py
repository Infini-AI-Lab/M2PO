import os
import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def make_prefix(dp, template_type):
    question = dp['problem']
    prefix = f"""Please solve the following math problem: {question}. The assistant first thinks about the reasoning process step by step and then provides the user with the answer. Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step. """
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/amc24')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    args = parser.parse_args()

    data_source = 'rawsh/2024_AMC12'
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    test_dataset = dataset['train']  # 2024_AMC12 uses "train" as its only split

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = example.pop('answer')
            # print(question, solution)
            # print('========================================')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Length of processed data: {len(test_dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
