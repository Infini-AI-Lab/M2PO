# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the math dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    problem = dp['problem']
    # NOTE: also need to change reward_score/countdown.py
    prefix = f"""Please solve the following math problem: {problem}. The assistant first thinks about the reasoning process step by step and then provides the user with the answer. Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step. """
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/deepscaler_preview')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    data_source = 'agentica-org/DeepScaleR-Preview-Dataset'

    # dataset = datasets.load_dataset(data_source, 'all', trust_remote_code=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train'].shuffle(seed=args.seed)
    if args.num_samples:
        train_dataset = train_dataset.select(range(args.num_samples))
    else:
        train_dataset = train_dataset.select(range(int(args.subset_ratio * len(train_dataset))))
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)

            solution = example.pop('answer')
            data = {
                "data_source": 'dapo_math',
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

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.num_samples:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.num_samples}_samples.parquet'))
    else:
        if args.subset_ratio == 1.0:
            train_dataset.to_parquet(os.path.join(local_dir, f'train.parquet'))
        else:
            train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.subset_ratio}.parquet'))

    # print data source and length
    print(f"Data source: {data_source}")
    print(f"Length of train dataset: {len(train_dataset)}")

    for i in range(5):
        print(f"Question: {train_dataset[i]['prompt'][0]['content']}")
        print(f"Solution: {train_dataset[i]['reward_model']['ground_truth']}")
        print("-" * 100)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)