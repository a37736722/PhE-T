import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if downsample:
            self.downsample_layers = nn.Sequential(
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        if self.downsample:
            identity = self.downsample_layers(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet18D1D(nn.Module):
    def __init__(self, input_shape=(1000,)):
        super().__init__()
        self.in_channels = 64
        
        # Input stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        
        # ResNet blocks
        self.block1 = ResBlock(64, 64)
        self.block2 = ResBlock(64, 64)
        self.block3 = ResBlock(64, 128, downsample=True)
        self.block4 = ResBlock(128, 128)
        self.block5 = ResBlock(128, 256, downsample=True)
        self.block6 = ResBlock(256, 256)
        self.block7 = ResBlock(256, 512, downsample=True)
        self.block8 = ResBlock(512, 512)
        
        # Output layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 128)
    
    def forward(self, x):
        x = self._pad(x)
        x = self.stem(x.float())
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _pad(self, x):
        target_length = 1024
        current_length = x.shape[-1]
        
        if current_length >= target_length:
            return x
        
        pad_size = target_length - current_length
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        
        return F.pad(x, (pad_left, pad_right))
    
    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        model = cls()
        model.load_state_dict(state_dict)    
        return model