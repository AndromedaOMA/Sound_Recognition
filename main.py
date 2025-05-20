import torch
import torchaudio

from configs.dataset_configs import DatasetConfigs
from dataset.urban_sound_8k import UrbanSoundDataset
from model.sound_recognition_model import SoundRecognitionModel
from configs.model_configs import ModelConfigs

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(input)
        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    model_configs = ModelConfigs()
    model = SoundRecognitionModel(model_configs).to(device)
    # state_dict = torch.load("train/SoundRecognitionModel.pth")
    state_dict = torch.load("train/best_model.pth")
    model.load_state_dict(state_dict)

    dataset_configs = DatasetConfigs()
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

    input, target = usd[0][0], usd[0][1]  # tensor(no_channels, fr, time)
    input.unsqueeze_(0)  # tensor(batch_size, no_channels, fr, time)

    predicted, expected = predict(model, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
