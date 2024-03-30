import os
import math
import torch
import random
import torchaudio
from typing import List, Optional, Tuple, Union

class Compose(object):
    """Compose several preprocess together.
    """

    def __init__(self, preprocess):
        """__init__.
        Args:
            preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
        """
        self.preprocess = preprocess

    def __call__(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): input audio waveform.
        Returns:
            torch.Tensor: the preprocessed audio waveform.
        """
        for p in self.preprocess:
            if p is not None:
                audio_data = p(audio_data)
        return audio_data

class FunctionalModule(torch.nn.Module):
    """Compose several preprocess together.
    """
    def __init__(self, functional):
        """
        Args:
            functional (function): function to convert in a torch.Module.
        """
        super().__init__()
        self.functional = functional

    def forward(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): input audio waveform.
        Returns:
            torch.Tensor: the preprocessed audio waveform.
        """
        return self.functional(audio_data)

class NormaliseUtterance(object):
    """Normalize per raw audio by removing the mean and divided by the standard deviation.
    """
    def __init__(self, eps: float = 1.0e-20):
        """__init__.
        Args:
            eps (float): epsilon coefficient to avoid division by zero
        """
        self.eps = eps

    def __call__(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): audio waveform to normalise.
        Returns:
            torch.Tensor: the normalised audio waveform.
        """

        return (audio_data - audio_data.mean()) / (audio_data.std() + self.eps)

class AddNoise(object):
    """Adding noise to an audio waveform.
       This code is mainly based on that implemented in https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, noise_path: str, sample_rate: float = 16000, snr_target: int = None):
        """__init__.

        Args:
            noise_path (str): the path where the noisy audio waveform is stored.
            sample_rate (float): the sample rate of the audio waveform.
            snr_target (int): a fixed signal-noise-rate value.
        """

        if not os.path.exists(noise_path):
            raise IOError(f'Noise path `{noise_path}` does not exist')

        # -- converting to mono-channel audio, re-sampling, and normalising the noisy audio waveform
        effects = [
            ["remix", "1"],
            ["rate", str(sample_rate)],
        ]
        self.entire_noise, noise_sr = torchaudio.sox_effects.apply_effects_file(noise_path, effects, normalize=False)
        self.entire_noise = torchaudio.functional.resample(self.entire_noise, orig_freq=noise_sr, new_freq=sample_rate)
        self.entire_noise_length = self.entire_noise.shape[-1]
        self.sample_rate = sample_rate
        self.snr_target = snr_target

    def __call__(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): audio waveform to which add noise.
        Returns:
            torch.Tensor: the noise-added audio waveform.
        """
        audio_length = audio_data.shape[-1]
        if self.entire_noise_length > audio_length:
            offset = random.randint(0, self.entire_noise_length - audio_length)
            noise = self.entire_noise[..., offset:offset+audio_length]
        elif self.entire_noise_length < audio_length:
            noise = torch.cat([self.entire_noise, torch.zeros((self.entire_noise.shape[0], audio_length - self.entire_noise_length))], dim=-1)
        noise_length = noise.shape[-1]

        snr_db = random.choice([-5, 0, 5, 10, 15, 20, 9999]) if not self.snr_target else self.snr_target
        # -- no applying data augmentation
        if snr_db == 9999:
            return audio_data

        # -- adding noise to the audio waveform
        snr = 10 ** (snr_db / 10.0)
        snr = snr ** 0.5

        audio_power = (audio_data ** 2).sum() / (audio_length * 1.0)
        noise_power = (noise ** 2).sum() / (noise_length * 1.0)

        noise = 1 / snr * noise * torch.sqrt(audio_power / noise_power)
        noisy_audio_data = audio_data + noise

        # snr = math.exp(snr_db / 10)
        # audio_power = audio_data.norm(p=2)
        # noise_power = noise.norm(p=2)
        # scale = snr * noise_power / audio_power
        # noisy_audio_data = (scale * audio_data + noise) / 2

        return noisy_audio_data

class SpeedRate(object):
    """Subsample/Upsample the number of frames of the audio waveform.
       This code is mainly based on that implemented in https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate: float = 16000):
        """__init__.

        Args:
            sample_rate (float): the sample rate of the audio waveform.
        """
        self.sample_rate = sample_rate

    def __call__(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): audio waveform to be re-sampled.
        Returns:
            torch.Tensor: re-sampled audio waveform.
        """
        speed_factor = random.choice([0.9, 1.0, 1.1])
        # -- no applying data augmentation
        if speed_factor == 1.0:
            return audio_data

        # -- re-sampling the audio waveform
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        resampled_audio_data, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data,
            self.sample_rate,
            sox_effects
        )

        return resampled_audio_data

class TimeMasking(object):
    """Apply the time-masking technique over an audio waveform.
       This code is mainly based on that implemented in https://github.com/facebookresearch/WavAugment
    """
    def __init__(self, sample_rate: float =16000, max_frames: Optional[int] = None, max_seconds: Optional[float] = None):
        """__init__.

        Args:
            sample_rate (float): the sample rate of the audio waveform.
            max_frames (int): the maximum number of acoustic frames to mask.
            max_seconds (int): the maximum number of seconds to mask.
        """

        assert max_frames or max_seconds
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.max_seconds = max_seconds

    def __call__(self, audio_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_data (torch.Tensor): audio waveform to which apply the time-masking technique
        Returns:
            torch.Tensor: masked audio waveform.
        """
        if self.max_frames is None:
            max_frames = int(self.sample_rate * self.max_seconds)
        else:
            max_frames = self.max_frames

        audio_length = audio_data.shape[-1]
        audio_mean_frame = audio_data.mean(dim=1)
        masked_audio_data = audio_data.detach().clone()
        for second in range(0, audio_length, self.sample_rate):
            mask_length = random.randint(0, max_frames)
            # -- applying masking
            if mask_length > 0:
                offset = random.randint(second + max_frames, second + self.sample_rate - mask_length)
                masked_audio_data[:, offset:offset+mask_length, ...] = audio_mean_frame

        return masked_audio_data

if __name__ == "__main__":
    x, sr = torchaudio.load("./wav-prueba.wav")
    an = AddNoise(noise_path="./noise/babble_noise.wav", sample_rate=16000, snr_target=5)
    y = an(SpeedRate()(x))
    torchaudio.save("./averquetal.wav", y, 16000)
"""
## Numpy implementation for adding noise to an audio waveform based on the Ma et al.'s code ##
class AddNoise(object):

    def __init__(self, noise, snr_target=None, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        self.noise = noise
        self.snr_levels = snr_levels
        self.snr_target = snr_target

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels) if not self.snr_target else self.snr_target
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal
"""
