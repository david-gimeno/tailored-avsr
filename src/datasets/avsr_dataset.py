import numpy as np
import pandas as pd
from unidecode import unidecode

import torch
import torchaudio
from torch.utils.data import Dataset

class AVSRDataset(Dataset):
    """Dataset to load different databases for audio- and video-only systems, as well as audio-visual ones.
    """

    def __init__(self, args, dataset_path, is_training=True):
        """__init__.

        Args:
            dataset_path (str): the path where CSV file defining a partition/dataset is stored.
            is_training (bool): indicates if is a dataset for training or evaluation purposes.
        """
        self.args = args
        self.dataset_path = dataset_path.lower()

        # -- getting dataset
        dataset = pd.read_csv(dataset_path, delimiter=',')

        # -- frame length limitation based on the video length
        self.samples = dataset[dataset['nframes'] <= args.training_settings['nframes']] if is_training else dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {}
        sample['sample_id'] = self.samples.iloc[index]['sampleID']

        if self.args.task in ['asr', 'avsr']:
            sample['audio'] = self.__get_audio_sample__(index)

        if self.args.task in ['vsr', 'avsr']:
            sample['video'] = self.__get_video_sample__(index)

        sample['transcription'] = self.__get_transcription_sample__(index)

        return sample

    def __get_audio_sample__(self, index):
        wav_path = self.samples.iloc[index]['wav_path']
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)

        return waveform # (T,)

    def __get_video_sample__(self, index):
        database = self.samples.iloc[index]['database']
        lips_path = self.samples.iloc[index]['lips_path']
        lips = np.load(lips_path)['data']

        # -- special case where videos were recorded at 50 fps
        if database.lower() == 'vlrf':
            lips = lips[::2, :, :]

        return torch.from_numpy(lips) # (T,96,96)

    def __get_transcription_sample__(self, index):
        transcription_path = self.samples.iloc[index]['transcription_path']
        transcription = open(transcription_path, 'r').readlines()[0].strip()

        # -- special case for english datasets
        transcription = transcription.upper().replace('{', '').replace('}', '')

        return transcription # (L,)
