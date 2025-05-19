import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.dataset_configs import DatasetConfigs
from configs.model_configs import ModelConfigs
from dataset.urban_sound_8k import UrbanSoundDataset
from model.sound_recognition_model import SoundRecognitionModel


def get_fold_indices(df, fold):
    train_idx = df[df['fold'] != fold].index.tolist()
    test_idx = df[df['fold'] == fold].index.tolist()
    return train_idx, test_idx


def check_acc(data_loader, model):
    no_correct = 0
    no_samples = 0
    model.eval()

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            predicted_labels = prediction.argmax(dim=1)

            no_correct += (predicted_labels == target).sum().item()
            no_samples += target.size(0)

    train_acc = no_correct / no_samples
    model.train()
    return train_acc


def train_per_epoch(model, data_loader, loss_fn, optimizer, device):
    for _, (input, target) in enumerate(tqdm(data_loader)):
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = check_acc(data_loader, model)

    print(f"Loss: {loss.item():.2f}%, Train Accuracy: {train_acc * 100:.2f}%")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    patience = 10
    best_acc = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        train_per_epoch(model, data_loader, loss_fn, optimizer, device)

        acc = check_acc(data_loader, model)
        if acc > best_acc:
            best_acc = acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print("===" * 10)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    dataset_configs = DatasetConfigs()
    model_configs = ModelConfigs()

    model = SoundRecognitionModel(model_configs).to(device)
    print(f"SoundRecognitionModel:\n{model}")

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

    accuracies = []

    # train_dataloader = create_data_loader(usd, model_configs.batch_size)
    for idx_fold in range(1, 11):
        print(f"\n=== Test fold index: {idx_fold} ===")

        train_idx, test_idx = get_fold_indices(usd.annotations, idx_fold)

        train_subset = Subset(usd, train_idx)
        test_subset = Subset(usd, test_idx)

        train_loader = DataLoader(train_subset, batch_size=model_configs.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=model_configs.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), model_configs.learning_rate)

        train(model, train_loader, loss_fn, optimiser, device, model_configs.epochs)

        acc = check_acc(test_loader, model)
        accuracies.append(acc)

    # Results
    accuracies = np.array(accuracies)
    print(f"\nMean Accuracy: {accuracies.mean() * 100:.2f}%")
    print(f"Standard Deviation: {accuracies.std() * 100:.2f}%")

    torch.save(model.state_dict(), "SoundRecognitionModel.pth")
