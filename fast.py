
from datasets import load_dataset

def save_split_to_file(dataset, filename):
    """Saves a dataset split to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        for item in dataset:
            # Change the label as necessary. 
            file.write(f"{item['text']}\t{item['coarse_label']}\n")

# Load the TREC dataset
dataset = load_dataset("trec")

# Save the train split
train_split = dataset['train']
save_split_to_file(train_split, './data/train.txt')

# Save the test split
test_split = dataset['test']
save_split_to_file(test_split, './data/test.txt')

print("Dataset splits saved")
