import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.dataset_configs import DatasetConfigs
from dataset.urban_sound_8k import UrbanSoundDataset
from configs.model_configs import ModelConfigs
from model.sound_recognition_model import SoundRecognitionModel


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for _, (input, target) in enumerate(tqdm(data_loader)):
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print("===" * 10)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    dataset_configs = DatasetConfigs()
    model_configs = ModelConfigs()

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
    train_dataloader = create_data_loader(usd, model_configs.batch_size)

    model = SoundRecognitionModel(model_configs).to(device)
    print(f"SoundRecognitionModel:\n{model}")

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), model_configs.learning_rate)

    train(model, train_dataloader, loss_fn, optimiser, device, model_configs.epochs)

    torch.save(model.state_dict(), "SoundRecognitionModel.pth")
