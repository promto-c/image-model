import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)

        # Decoder
        self.decoder1 = self._conv_transpose_block(512, 256)
        self.decoder2 = self._conv_transpose_block(512, 128)
        self.decoder3 = self._conv_transpose_block(256, 64)
        self.decoder4 = self._conv_transpose_block(128, 32)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        dec1 = self.decoder1(enc4)
        dec2 = self.decoder2(torch.cat((self._resize(dec1, enc3), enc3), dim=1))
        dec3 = self.decoder3(torch.cat((self._resize(dec2, enc2), enc2), dim=1))
        dec4 = self.decoder4(torch.cat((self._resize(dec3, enc1), enc1), dim=1))

        # Final convolution
        output = self.final_conv(dec4)
        output = self.sigmoid(output)
        return output

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _conv_transpose_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def _resize(self, x, sample):
        if x.shape[2:] == sample.shape[2:]:
            return x

        return F.interpolate(x, size=sample.shape[2:], mode='bilinear', align_corners=True)