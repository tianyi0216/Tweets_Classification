## Data Loader

import pandas as pd
import torch
from torch.utils.data import Dataset
import prompts
from transformers import AutoTokenizer

class TweetDataset(Dataset):
    """
    A dataset class for the tweet classification task.
    """

    def __init__(self, df, tokenizer = None, train = False, prompt_type = "baseline"):
        """
        Initializes the dataset
        Args:
            csv_path: path to the csv file for the dataset
        """
        self.df = df

        self.train = train
        self.prompt_type = prompt_type
        if self.train:
            self.tokenizer = tokenizer
            self.label_map = {
                'FAVOR': 'in-favor',
                'AGAINST': 'against',
                'NONE': 'neutral-or-unclear'
            }

    def __len__(self):
        """
        Returns dataset size
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset given an index
        Args:
            idx: index of the item
        Returns:
            A dictionary containing the tweet and label
        """
        row = self.df.iloc[idx]
        tweet = row['tweet']
        label = row['label_majority']
        if not self.train:
            return {
                'tweet': tweet,
                'label': label
            }
        else:
            input_text = prompts.format_prompt(prompts.prompts[self.prompt_type], tweet)
            target_text = self.label_map[label]
            tokenized_input = self.tokenizer(input_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            tokenized_target = self.tokenizer(target_text, return_tensors="pt", padding=False, truncation=True, max_length=10)
            return {
                'input_ids': tokenized_input['input_ids'].flatten(),
                'attention_mask': tokenized_input['attention_mask'].flatten(),
                'labels': tokenized_target['input_ids'].flatten()
            }

if __name__ == "__main__":
    # run this file directly checks the dataset code
    df = pd.read_csv("data/Q2_20230202_majority.csv")
    dataset = TweetDataset(df)
    # check first batch
    print("Validating eval dataset:")
    print(dataset[0])
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    dataset_train = TweetDataset(df, train=True, tokenizer=tokenizer)
    print("Validating train dataset:")
    print(dataset_train[0])

