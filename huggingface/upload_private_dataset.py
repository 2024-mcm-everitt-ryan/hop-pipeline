#!/usr/bin/env python3
import sys
import os
from datasets import (load_dataset, DatasetDict)

from huggingface_hub import login

def main():
    if len(sys.argv) > 1:
        print("\nArguments passed to the script:")
        for arg in sys.argv[1:]:
            print(arg)

    # Parameters
    hf_site_id = sys.argv[1].strip() # Org, account, etc.
    dataset_name = sys.argv[2].strip()
    repo_id = f'{hf_site_id}/{dataset_name}'
    hf_token = os.environ.get(sys.argv[3].strip())
    dataset_file = f'{sys.argv[4]}'

    split = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
    val_size = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    test_size = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2

    print('Logging into Hugging Face')
    login(hf_token, add_to_git_credential=True)

    print(f'Loading data from: {dataset_file}')

    if split:
        # Dataset
        dataset = load_dataset('parquet', split='train', data_files=dataset_file)
        print(f'Number of samples: {len(dataset)}')
        print(f'Number of columns: {dataset.column_names}')

        print(f'Split the data with val_size={val_size} and test_size={test_size}.')
        dataset = dataset.train_test_split(test_size=test_size, seed=2024, shuffle=True)

        ds_train = dataset['train']
        ds_test = dataset['test']
        dataset = ds_train.train_test_split(test_size=val_size, seed=2024, shuffle=False)
        ds_train = dataset['train']
        ds_val = dataset['test']
        del dataset

        print(ds_train)
        print(ds_val)
        print(ds_test)

        ds = DatasetDict()
        ds["train"] = ds_train
        ds["val"] = ds_val
        ds["test"] = ds_test

        print(f'Uploading dataset to {repo_id}...')
        ds.push_to_hub(repo_id,  private=True, token=hf_token)
    else:
        # Dataset
        dataset = load_dataset('parquet', data_files=dataset_file)
        print(f'Number of samples: {len(dataset["train"])}')
        print(f'Number of columns: {dataset["train"].column_names}')
        print(f'Uploading dataset to {repo_id}...')
        dataset.push_to_hub(repo_id,  private=True, token=hf_token)

if __name__ == "__main__":
    main()
