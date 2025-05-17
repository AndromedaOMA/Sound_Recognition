import torch
import torchaudio
from configs.dataset_configs import DatasetConfigs
from dataset.urban_sound_8k import UrbanSoundDataset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    dataset_configs = DatasetConfigs()

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
