import re
# build corpus


# dataset = 'aclImdb'

# f = open('data/' + dataset + '.txt', 'r')
# lines = f.readlines()
# docs = []
# for line in lines:
#     temp = line.split("\t")
#     doc_file = open(temp[0], 'r')
#     doc_content = doc_file.read()
#     doc_file.close()
#     print(temp[0], doc_content)
#     doc_content = doc_content.replace('\n', ' ')
#     docs.append(doc_content)


# corpus_str = '\n'.join(docs)
# f.close()

# f = open('data/corpus/' + dataset + '.txt', 'w')
# f.write(corpus_str)
# f.close()



# # datasets from PTE paper
import torch_geometric
from torch_geometric.datasets import DBLP
import os
import torch
# Download and load the DBLP dataset from PyTorch Geometric
dataset = DBLP(root='/tmp/DBLP')

# The dataset contains paper-author, paper-conference, and paper-term relationships
# along with labels for each paper.

# Extracting papers, authors, and labels
papers, authors, labels = dataset[0].p_nodes, dataset[0].a_nodes, dataset[0].y

# Create a directory to store the processed files
os.makedirs('data/dblp', exist_ok=True)

# Assuming you have a way to split the dataset into train and test
# For this example, let's split it manually (this is just an example, you should use a proper train-test split)
train_size = int(0.8 * len(labels))
test_size = len(labels) - train_size

train_mask = torch.zeros(len(labels), dtype=torch.bool)
test_mask = torch.zeros(len(labels), dtype=torch.bool)
train_mask[:train_size] = True
test_mask[train_size:] = True

# Write train labels to file
with open('data/dblp/label_train.txt', 'w') as f:
    for i in range(train_size):
        label = labels[i].item()
        f.write(f'{label}\n')

# Write test labels to file
with open('data/dblp/label_test.txt', 'w') as f:
    for i in range(train_size, len(labels)):
        label = labels[i].item()
        f.write(f'{label}\n')

# Processing the labels as per your requirement
doc_id = 0
doc_name_list = []

# Process train labels
with open('data/dblp/label_train.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        string = f'{doc_id}\ttrain\t{line.strip()}'
        doc_name_list.append(string)
        doc_id += 1

# Process test labels
with open('data/dblp/label_test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        string = f'{doc_id}\ttest\t{line.strip()}'
        doc_name_list.append(string)
        doc_id += 1

# Writing the final processed data
doc_list_str = '\n'.join(doc_name_list)
with open('data/dblp.txt', 'w') as f:
    f.write(doc_list_str)

# # TREC, R8, R52, WebKB

# dataset = 'aclImdb'

# f = open('data/' + dataset + '/train.txt', 'r')
# lines = f.readlines()
# f.close()

# doc_id = 0
# doc_name_list = []
# doc_content_list = []

# for line in lines:
#     line = line.strip()
#     label = line[:line.find('\t')]
#     content = line[line.find('\t') + 1:]
#     string = str(doc_id) + '\t' + 'train' + '\t' + label
#     doc_name_list.append(string)
#     doc_content_list.append(content)
#     doc_id += 1

# f = open('data/' + dataset + '/test.txt', 'r')
# lines = f.readlines()
# f.close()

# for line in lines:
#     line = line.strip()
#     label = line[:line.find('\t')]
#     content = line[line.find('\t') + 1:]
#     string = str(doc_id) + '\t' + 'test' + '\t' + label
#     doc_name_list.append(string)
#     doc_content_list.append(content)
#     doc_id += 1

# doc_list_str = '\n'.join(doc_name_list)

# f = open('data/' + dataset + '.txt', 'w')
# f.write(doc_list_str)
# f.close()

# doc_name_list_str = '\n'.join(doc_name_list)

# f = open('data/' + dataset + '.txt', 'w')
# f.write(doc_list_str)
# f.close()

# doc_content_list_str = '\n'.join(doc_content_list)

# f = open('data/corpus/' + dataset + '.txt', 'w')
# f.write(doc_content_list_str)
# f.close()
