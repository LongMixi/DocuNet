import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUNet(nn.Module):
    """
    UNet for global reasoning over attention maps.
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super().__init__()

        down_channel = kwargs.get('down_channel', 256)

        down_channel_2 = down_channel * 2
        up_channel_1 = down_channel_2 * 2
        up_channel_2 = down_channel * 2

        self.inc = InConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_2)

        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)

        self.outc = OutConv(up_channel_2 // 4, class_number)

    def forward(self, attention_channels):
        """
        Input: batch_size x channels x width x height
        Output: batch_size x width x height x class
        """
        x = attention_channels

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        output = self.outc(x)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


class DoubleConv(nn.Module):
    """ (Conv → BN → ReLU) * 2 """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            # safer for PyTorch 2.x
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        # After concatenation: channels_x2 → DoubleConv expects (in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Compute padding
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # F.pad uses (left, right, top, bottom)
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
