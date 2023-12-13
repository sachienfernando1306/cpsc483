# from datasets import load_dataset
# import fasttext
# import pandas as pd
# import os

# # Load the dataset from Hugging Face datasets
# dataset = load_dataset("dbpedia_14")

# def prepare_fasttext_dataset(dataset, split, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for item in dataset[split]:
#             label = '__label__' + str(item['label'])
#             text = item['content'].replace('\n', ' ')
#             f.write(label + ' ' + text + '\n')

# prepare_fasttext_dataset(dataset, 'train', 'dbpedia_train.txt')
# prepare_fasttext_dataset(dataset, 'test', 'dbpedia_test.txt')


# model = fasttext.train_supervised(input='dbpedia_train.txt')
# model.save_model("dbpedia_model.bin")

# def print_results(N, p, r):
#     print("N\t" + str(N))
#     print("P@{}\t{:.3f}".format(1, p))
#     print("R@{}\t{:.3f}".format(1, r))

# # Evaluate the model
# print_results(*model.test('dbpedia_test.txt'))

from datasets import load_dataset

def save_split_to_file(dataset, filename):
    """Saves a dataset split to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        for item in dataset:
            # Assuming the dataset has 'text' and 'label' firmelds
            file.write(f"{item['text']}\t{item['label']}\n")

# Load the TREC dataset
dataset = load_dataset("ag_news")

# Save the train split
train_split = dataset['train']
save_split_to_file(train_split, 'train.txt')

# Save the test split
test_split = dataset['test']
save_split_to_file(test_split, 'test.txt')

print("Dataset splits saved")
