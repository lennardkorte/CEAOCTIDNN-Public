import torch
import torch.nn as nn
#from torchsummary import summary


def get_configs(arch):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")

class ResNetAutoEncoder(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck, mirror, freeze_enc, depth):

        super(ResNetAutoEncoder, self).__init__()

        self.encoder = ResNetEncoder(layer_cfg=layer_cfg, version=version, bottleneck=bottleneck, depth=depth)
        if freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if mirror:
            self.decoder = ResNetDecoder(layer_cfg=layer_cfg[::-1], bottleneck=bottleneck, conv_trans=True, depth=depth)
        else:    
            self.decoder = ResNetDecoder(layer_cfg=[3, 4, 6, 3][::-1], bottleneck=True, conv_trans=False, depth=depth)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ResNet(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck, num_classes, dropout):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(layer_cfg=layer_cfg, version=version, bottleneck=bottleneck, depth=5)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=2048, out_features=num_classes)
            )
            
        else:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=512, out_features=num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # alternative to pytorch version below
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        '''
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        '''

        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
        # This variant is also known as ResNet V1.5 and improves accuracy according to
        # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck, depth):
        super(ResNetEncoder, self).__init__()
        self.depth = depth

        if len(layer_cfg) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:
            if self.depth >= 2:
                self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=layer_cfg[0], downsample_method="pool", version=version)
            if self.depth >= 3:
                self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=layer_cfg[1], downsample_method="conv", version=version)
            if self.depth >= 4:
                self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=layer_cfg[2], downsample_method="conv", version=version)
            if self.depth >= 5:
                self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=layer_cfg[3], downsample_method="conv", version=version)

        else:
            if self.depth >= 2:
                self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=layer_cfg[0], downsample_method="pool", version=version)
            if self.depth >= 3:
                self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=layer_cfg[1], downsample_method="conv", version=version)
            if self.depth >= 4:
                self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=layer_cfg[2], downsample_method="conv", version=version)
            if self.depth >= 5:
                self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=layer_cfg[3], downsample_method="conv", version=version)

    def forward(self, x):

        x = self.conv1(x)
        if self.depth >= 2:
            x = self.conv2(x)
        if self.depth >= 3:
            x = self.conv3(x)
        if self.depth >= 4:
            x = self.conv4(x)
        if self.depth >= 5:
            x = self.conv5(x)

        return x

class ResNetDecoder(nn.Module):

    def __init__(self, layer_cfg, bottleneck, conv_trans, depth, in_channels=[2048, 1024, 512, 256, 64]):
        super(ResNetDecoder, self).__init__()
        self.depth = depth

        if len(layer_cfg) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:
            if self.depth >= 5:
                self.conv1 = DecoderBottleneckBlock(in_channels=in_channels[0], hidden_channels=min(512, in_channels[1]), down_channels=in_channels[1], layers=layer_cfg[0], conv_trans=conv_trans)
            if self.depth >= 4:
                self.conv2 = DecoderBottleneckBlock(in_channels=in_channels[1], hidden_channels=min(256, in_channels[2]), down_channels=in_channels[2],  layers=layer_cfg[1], conv_trans=conv_trans)
            if self.depth >= 3:
                self.conv3 = DecoderBottleneckBlock(in_channels=in_channels[2],  hidden_channels=min(128, in_channels[3]), down_channels=in_channels[3],  layers=layer_cfg[2], conv_trans=conv_trans)
            if self.depth >= 2:
                self.conv4 = DecoderBottleneckBlock(in_channels=in_channels[3],  hidden_channels=min(64, in_channels[4]),  down_channels=in_channels[4],   layers=layer_cfg[3], conv_trans=conv_trans)

        else:
            if self.depth >= 5:
                self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=layer_cfg[0], conv_trans=conv_trans)
            if self.depth >= 4:
                self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=layer_cfg[1], conv_trans=conv_trans)
            if self.depth >= 3:
                self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=layer_cfg[2], conv_trans=conv_trans)
            if self.depth >= 2:
                self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=layer_cfg[3], conv_trans=conv_trans)

        if conv_trans:
            self.conv5 = nn.Sequential(
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
            )
        else:
            self.conv5 = nn.Sequential(
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False),
            )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        if self.depth >= 5:
            x = self.conv1(x)
        if self.depth >= 4:
            x = self.conv2(x)
        if self.depth >= 3:
            x = self.conv3(x)
        if self.depth >= 2:
            x = self.conv4(x)
        
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method, version):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, version=version, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method, version):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True, version=version)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers, conv_trans):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True, conv_trans=conv_trans)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False, conv_trans=conv_trans)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers, conv_trans):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True, conv_trans=conv_trans)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False, conv_trans=conv_trans)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample, version):
        super(EncoderResidualLayer, self).__init__()

        self.version = version

        if downsample:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                )

        else:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                )

        if version != 2.0:
            self.weight_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                
            )

        if downsample:
            if version != 2.0:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    
                )
        else:
            self.downsample = None

        if version != 2.0:
            self.relu = nn.Sequential(
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        if self.version != 2.0:
            x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample, version):
        super(EncoderBottleneckLayer, self).__init__()
        
        self.version = version

        if downsample:
            if self.version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                )
        else:
            if self.version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                )

        if self.version != 2.0:
            self.weight_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if self.version != 2.0:
            self.weight_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )

        if downsample:
            if self.version != 2.0:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=up_channels),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                )

        elif (in_channels != up_channels):
            self.downsample = None
            if self.version != 2.0:
                self.up_scale = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=up_channels),
                )
            else:
                self.up_scale = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                )

        else:
            self.downsample = None
            self.up_scale = None

        if self.version != 2.0:
            self.relu = nn.Sequential(
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        if self.version != 2.0:
            x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample, conv_trans):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            if conv_trans:
                self.weight_layer2 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),              
                )
            else:
                self.weight_layer2 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),              
                )

        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            if conv_trans:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
                )
            else:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=1, bias=False) 
                )

        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample, conv_trans):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            if conv_trans:
                self.weight_layer3 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
                )
            else:
                self.weight_layer3 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, bias=False)
                )
    
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            if conv_trans:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
                )
            else:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, bias=False),
                )

        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x

if __name__ == "__main__":

    input = torch.randn((5,3,224,224))
    print(input.shape)

    version = 2.0
    layer_cfg, bottleneck = get_configs("resnet50")

    encoder = ResNetEncoder(layer_cfg, version, bottleneck, 5)
    output = encoder(input)
    print(output.shape)

    decoder = ResNetDecoder(layer_cfg[::-1], bottleneck, conv_trans=False, depth=5)
    output = decoder(output)
    print(output.shape)

    autenc = ResNetAutoEncoder(layer_cfg, version=version, bottleneck=bottleneck, mirror=False, freeze_enc=True, depth=5)
    output = autenc(input)
    print(output.shape)

    model = ResNet(layer_cfg, version, bottleneck, num_classes=2, dropout=0.0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Training Parameters', pytorch_total_params)

    #from torchsummary import summary
    #summary(encoder)
    #summary(decoder)
    #summary(autenc)
    #summary(model)