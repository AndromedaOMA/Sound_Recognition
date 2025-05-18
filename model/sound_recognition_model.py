import json

from torch import nn
from torchsummary import summary
from configs.model_configs import ModelConfigs


class SoundRecognitionModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(configs.in_channels, configs.mid_channels, configs.kernel_size, configs.stride, configs.padding),
            nn.BatchNorm2d(num_features=configs.mid_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=configs.p_kernel1, stride=configs.p_stride1),

            nn.Conv2d(configs.mid_channels, configs.mid_channels, configs.kernel_size, configs.stride, configs.padding),
            nn.BatchNorm2d(num_features=configs.mid_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=configs.p_kernel2, stride=configs.p_stride2),

            nn.Conv2d(configs.mid_channels, configs.mid_channels, configs.kernel_size, configs.stride, configs.padding),
            nn.BatchNorm2d(num_features=configs.mid_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=configs.p_kernel2, stride=configs.p_stride2)
        )
        conv_out_size = 4 * 2 * configs.mid_channels  # shape signal: torch.Size([1, 64, 44])
        self.fc_block = nn.Sequential(
            nn.Linear(conv_out_size, configs.mid_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(configs.mid_channels, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, batch):
        batch = self.conv_block(batch)
        batch = nn.Flatten()(batch)
        batch = self.fc_block(batch)
        return batch


if __name__ == "__main__":
    model_configs = ModelConfigs()
    with open("model_config.json", mode="w", encoding="utf-8") as file:
        json.dump(model_configs.__dict__, file, indent=4)
    model = SoundRecognitionModel(model_configs).cuda()
    summary(model, (1, 64, 44))
