import numpy as np
import pandas as pd
from unidecode import unidecode

import torch
import torchaudio
from torch.utils.data import Dataset

class MyVideoDataset(Dataset):
    """Dataset to load different databases for the video-only modality.
    """

    def __init__(self, dataset_path, nframes, is_training=True):
        """__init__.

        Args:
            dataset_path (str): the path where CSV file defining a partition/dataset is stored.
            nframes (int): only samples with a video-frame length lower than this value are considered.
            is_training (bool): indicates if is a dataset for training or evaluation purposes.
        """
        self.dataset_path = dataset_path.lower()

        # -- getting dataset
        dataset = pd.read_csv(dataset_path, delimiter=",")

        # -- frame length limitation based on the video length
        self.samples = dataset[dataset["nframes"]<=nframes] if is_training else dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        lips = self.__get_visual_sample__(index)
        transcription = self.__get_transcription_sample__(index)

        return lips, transcription

    def __get_visual_sample__(self, index):
        database = self.samples.iloc[index]["database"]
        lips_path = self.samples.iloc[index]["lips_path"]
        lips = np.load(lips_path)["data"]
        # -- special case where videos were recorded at 50 fps
        if database.lower() == "vlrf":
            lips = lips[::2, :, :]

        return torch.from_numpy(lips) # (T,96,96)

    def __get_transcription_sample__(self, index):
        transcription_path = self.samples.iloc[index]["transcription_path"]
        transcription = open(transcription_path, "r").readlines()[0].strip()

        # -- special cases
        # if ("liprtve" in self.dataset_path) or ("vlrf" in self.dataset_path):
        #     transcription.lower()
        #     transcription = re.sub(r"[^\w\s]","",transcription)
        #     transcription = unidecode(transcription.replace("ñ", "N")).replace("N", "ñ")

        transcription = transcription.upper().replace("{", "").replace("}", "")

        return transcription
