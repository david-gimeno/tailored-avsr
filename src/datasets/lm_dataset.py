import numpy as np
import pandas as pd
from unidecode import unidecode

import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    """Dataset to load different databases for Language Model training.
    """

    def __init__(self, dataset_path, from_dataset_partition=True):
        """__init__.

        Args:
            dataset_path (str): the path where CSV file defining a partition/dataset is stored.
            from_dataset_partition (bool): indicates if the text is taken from a audio-visual dataset or, on the contrary, from a raw text file
        """
        self.dataset_path = dataset_path
        self.from_dataset_partition = from_dataset_partition

        # -- getting dataset
        if self.from_dataset_partition:
            self.samples = pd.read_csv(dataset_path, delimiter=",")
        else:
            self.samples = [l.strip() for l in open(dataset_path, "r", encoding = "utf-8").readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.__get_text_sample__(index)

        return text

    def __get_text_sample__(self, index):
        if self.from_dataset_partition:
            text_path = self.samples.iloc[index]["transcription_path"]
            text = open(text_path, "r").readlines()[0].strip()
        else:
            text = self.samples[index]

        text = text.upper().replace("{", "").replace("}", "")

        return text
