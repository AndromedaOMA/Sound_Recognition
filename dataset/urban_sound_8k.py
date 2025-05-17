import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os

import json
from configs.dataset_configs import DatasetConfigs


class UrbanSoundDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_ample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal -> tensor(no_channels, samples) -> tensor(2, 16000)
        signal = self._resample_if_necessary(signal, sample_rate)  # Uniform Sample Rate
        signal = self._mix_down_if_necessary(signal)  # Uniform no. of channels
        # signal -> tensor(no_channels, samples) -> tensor(1, 16000)
        signal = self._cut_if_necessary(signal)
        # signal -> tensor(no_channels, samples) -> tensor(1, num_samples)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)  # MelSpectrogram
        return signal, label

    def _get_audio_sample_path(self, index):
        folder = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])
        return path

    def _get_audio_ample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _resample_if_necessary(self, signal, sr):  # https://docs.pytorch.org/audio/master/generated/torchaudio.transforms.Resample.html#torchaudio.transforms.Resample
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal -> tensor(no_channels, samples) -> tensor(2, 16000) -> tensor(1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        # signal -> tensor(no_channels, samples) -> tensor(1, 16000) -> tensor(1, num_samples) by cutting the surplus
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        # signal -> tensor(no_channels, samples) -> tensor(1, 16000) -> tensor(1, num_samples) by padding the less data
        if signal.shape[1] < self.num_samples:
            # signal = torch.nn.functional.pad(signal, (0, self.num_samples - signal.shape[1]), mode='constant', value=0)
            signal = torch.nn.functional.pad(signal, (0, self.num_samples - signal.shape[1]))
        return signal


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    dataset_configs = DatasetConfigs()
    with open("dataset_config.json", mode="w", encoding="utf-8") as file:
        json.dump(dataset_configs.__dict__, file, indent=4)

    # print(f"dataset_configs.ANNOTATIONS_FILE: {dataset_configs.ANNOTATIONS_FILE},\n"
    #       f"dataset_configs.AUDIO_DIR: {dataset_configs.AUDIO_DIR},\n"
    #       f"dataset_configs.SAMPLE_RATE: {dataset_configs.SAMPLE_RATE},\n"
    #       f"dataset_configs.NUM_SAMPLES: {dataset_configs.NUM_SAMPLES}\n")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=dataset_configs.SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(dataset_configs.ANNOTATIONS_FILE,
                            dataset_configs.AUDIO_DIR,
                            mel_spectrogram,
                            dataset_configs.SAMPLE_RATE,
                            dataset_configs.NUM_SAMPLES,
                            device)

    print(f"There are {len(usd)} samples!")

    signal, label = usd[0]
    print(f"signal: {signal}\nlabel: {label}")
