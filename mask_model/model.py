import torch
import torch.nn as nn
import torchvision.models as models


class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.in_channels = in_channels

        # Position attention module
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Channel attention module
        self.beta = nn.Parameter(torch.zeros(1))

        # Final fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def position_attention(self, x):
        b, c, h, w = x.size()
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x

    def channel_attention(self, x):
        b, c, h, w = x.size()
        x_reshape = x.view(b, c, -1)
        energy = torch.bmm(x_reshape, x_reshape.permute(0, 2, 1))
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(attention, x_reshape)
        out = out.view(b, c, h, w)
        return self.beta * out + x

    def forward(self, x):
        position_out = self.position_attention(x)
        channel_out = self.channel_attention(x)
        out = torch.cat([position_out, channel_out], dim=1)
        out = self.fusion_conv(out)
        return out


class ResNetSkeleton(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetSkeleton, self).__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.resnet18 = nn.Sequential(*list(resnet.children())[:-2])
        self.layer0 = nn.Sequential(*list(self.resnet18.children())[:3])
        self.layer1 = nn.Sequential(*list(self.resnet18.children())[3:5])
        self.layer2 = self.resnet18[5]
        self.layer3 = self.resnet18[6]
        self.layer4 = self.resnet18[7]

        # Dilated convolutions for layers
        self.dilation_conv1_l2 = self._make_dilated_conv(128, 256, 2)
        self.dilation_conv2_l2 = self._make_dilated_conv(128, 256, 4)
        self.dilation_conv3_l2 = self._make_dilated_conv(128, 256, 8)
        self.cbam_l2 = DualAttention(256 * 3)

        self.dilation_conv1_l3 = self._make_dilated_conv(256, 512, 2)
        self.dilation_conv2_l3 = self._make_dilated_conv(256, 512, 4)
        self.dilation_conv3_l3 = self._make_dilated_conv(256, 512, 8)
        self.cbam_l3 = DualAttention(512 * 3)

        self.dilation_conv1_l4 = self._make_dilated_conv(512, 1024, 2)
        self.dilation_conv2_l4 = self._make_dilated_conv(512, 1024, 4)
        self.dilation_conv3_l4 = self._make_dilated_conv(512, 1024, 8)
        self.cbam_l4 = DualAttention(1024 * 3)

        # Upsampling path
        self.upsample1 = self._make_transpose_conv(3072, 512, 2)
        self.upsample2 = self._make_transpose_conv(2048, 512, 2)
        self.upsample3 = self._make_transpose_conv(1280, 256, 2)

        self.dilation_conv1_u3 = self._make_dilated_conv(256, 256, 2)
        self.dilation_conv2_u3 = self._make_dilated_conv(256, 256, 4)
        self.dilation_conv3_u3 = self._make_dilated_conv(256, 256, 8)

        self.upsample4 = self._make_transpose_conv(768, 128, 2)

        self.dilation_conv1_u4 = self._make_dilated_conv(128, 128, 2)
        self.dilation_conv2_u4 = self._make_dilated_conv(128, 128, 4)
        self.dilation_conv3_u4 = self._make_dilated_conv(128, 128, 8)

        self.upsample5 = self._make_transpose_conv(384, 64, 2)
        self.convf = nn.Conv2d(64, num_classes, kernel_size=1)
        self.mask_output = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def _make_dilated_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_transpose_conv(self, in_channels, out_channels, scale_factor):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=scale_factor,
                padding=0,
                output_padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        layer0 = self.layer0(img)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Apply dilation and attention to layers
        y1 = self.dilation_conv1_l2(layer2)
        y2 = self.dilation_conv2_l2(layer2)
        y3 = self.dilation_conv3_l2(layer2)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.cbam_l2(y)

        z1 = self.dilation_conv1_l3(layer3)
        z2 = self.dilation_conv2_l3(layer3)
        z3 = self.dilation_conv3_l3(layer3)
        z = torch.cat([z1, z2, z3], dim=1)
        z = self.cbam_l3(z)

        w1 = self.dilation_conv1_l4(layer4)
        w2 = self.dilation_conv2_l4(layer4)
        w3 = self.dilation_conv3_l4(layer4)
        w = torch.cat([w1, w2, w3], dim=1)
        w = self.cbam_l4(w)

        # Upsampling path
        x = self.upsample1(w)
        x = torch.cat([x, z], dim=1)
        x = self.upsample2(x)
        x = torch.cat([x, y], dim=1)
        x = self.upsample3(x)

        d1 = self.dilation_conv1_u3(x)
        d2 = self.dilation_conv2_u3(x)
        d3 = self.dilation_conv3_u3(x)
        x = torch.cat([d1, d2, d3], dim=1)

        x = self.upsample4(x)

        e1 = self.dilation_conv1_u4(x)
        e2 = self.dilation_conv2_u4(x)
        e3 = self.dilation_conv3_u4(x)
        x = torch.cat([e1, e2, e3], dim=1)

        x = self.upsample5(x)
        x = self.convf(x)
        output = self.mask_output(x)

        return torch.sigmoid(output)
