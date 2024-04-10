"""
fetch training dataset from huggingface
    - wikipedia
    - bookcorpus
"""

import datasets

def fetch_wikipedia():
    dataset = datasets.load_dataset('wikipedia', '20220301.en')
    return dataset

def fetch_bookcorpus():
    dataset = datasets.load_dataset('bookcorpus')
    return dataset

def fetch_dataset(dataset_name):
    if dataset_name == 'wikipedia':
        return fetch_wikipedia()
    elif dataset_name == 'bookcorpus':
        return fetch_bookcorpus()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")