## Data Loader

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TweetDataset(Dataset):
    """
    A dataset class for the tweet classification task.
    """

    def __init__(self, csv_path):
        """
        Initializes the dataset
        Args:
            csv_path: path to the csv file for the dataset
        """
        self.df = pd.read_csv(csv_path)

        # drop rows with missing values and duplicates
        self.df = self.df.dropna(subset=['tweet', 'label_majority'])
        self.df = self.df.drop_duplicates(subset=['tweet'], keep='first')

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
        
        return {
            'tweet': tweet,
            'label': label
        }
    
def get_dataloader(dataset, batch_size, shuffle=True, collate_fn=None):
    """
    Create a PyTorch DataLoader for the dataset
    Args:
        dataset: TweetDataset object
        batch_size: batch size
        shuffle: whether to shuffle the dataset
        collate_fn: function to collate the data for the loader
    Returns:
        A PyTorch DataLoader object for the TweetDataset
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    """
    Collate function for the dataset
    Args:
        batch: batch of data
    Returns:
        batch: batch of data
    """
    return batch

if __name__ == "__main__":
    # run this file directly checks the dataset code
    dataset = TweetDataset("data/Q2_20230202_majority.csv")
    dataloader = get_dataloader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
    

