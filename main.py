import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_ample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        folder = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])
        return path

    def _get_audio_ample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "E:/UrbanSound8k/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "E:/UrbanSound8k/audio"
    SAMPLE_RATE = 16000

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples!")

    signal, label = usd[0]
    print(f"signal: {signal}\nlabel: {label}")
