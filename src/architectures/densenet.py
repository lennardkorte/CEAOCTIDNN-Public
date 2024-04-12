import sys
import torch
import torch.nn as nn
#from torchsummary import summary
from pathlib import Path
script_directory = Path(__file__).parent.resolve()
if str(script_directory) not in sys.path:
    sys.path.append(str(script_directory))
import resnet

def get_configs(arch):

    # True or False means wether to use BottleNeck

    if arch == 'densenet121':
        return [6, 12, 24, 16]
    elif arch == 'densenet169':
        return [6, 12, 32, 32]
    elif arch == 'densenet201':
        return [6, 12, 48, 32]
    elif arch == 'densenet264':
        return [6, 12, 64, 48]
    else:
        raise ValueError("Undefined model")

class DenseNetAutoEncoder(nn.Module):

    def __init__(self, layer_cfg, mirror, depth, conv_trans, freeze_enc, k=32):

        super(DenseNetAutoEncoder, self).__init__()

        self.encoder = DenseNetEncoder(layer_cfg=layer_cfg, k=k, depth=depth)
        if freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if mirror:
            self.decoder = DenseNetDecoder(layer_cfg=layer_cfg[::-1], k=k, depth=depth, conv_trans=conv_trans)
        else:
            layer_cfg, bottleneck = resnet.get_configs("resnet50")
            self.decoder = resnet.ResNetDecoder(layer_cfg=layer_cfg[::-1], bottleneck=bottleneck, conv_trans=False, depth=depth, in_channels=[1024, 1024, 512, 256, 64])

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class DenseNetEncoder(nn.Module):

    def __init__(self, layer_cfg, k, depth):
        super(DenseNetEncoder, self).__init__()
        self.depth = depth

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2*k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=2*k),
            nn.ReLU(inplace=True),
        )

        if self.depth >= 2:
            num_features = 2*k
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.block1 = DenseBlockEncoder(input_features=2*k, layers=layer_cfg[0], k=k)
        
        if self.depth >= 3:
            num_features += layer_cfg[0]*k
            self.trans1 = TransitionEncoder(input_features=num_features, output_features=num_features//2)

            num_features = num_features // 2
            self.block2 = DenseBlockEncoder(input_features=num_features, layers=layer_cfg[1], k=k)

        if self.depth >= 4:
            num_features += layer_cfg[1]*k
            self.trans2 = TransitionEncoder(input_features=num_features, output_features=num_features//2)

            num_features = num_features // 2
            self.block3 = DenseBlockEncoder(input_features=num_features, layers=layer_cfg[2], k=k)

        if self.depth >= 5:
            num_features += layer_cfg[2]*k
            self.trans3 = TransitionEncoder(input_features=num_features, output_features=num_features//2)
        
            num_features = num_features // 2
            self.block4 = DenseBlockEncoder(input_features=num_features, layers=layer_cfg[3], k=k)
        
    def forward(self, x):

        x = self.conv1(x)

        if self.depth >= 2:
            x = self.maxpool(x)
            x = self.block1(x)
        
        if self.depth >= 3:
            x = self.trans1(x)
            x = self.block2(x)

        if self.depth >= 4:
            x = self.trans2(x)
            x = self.block3(x)

        if self.depth >= 5:
            x = self.trans3(x)
            x = self.block4(x)

        return x
    
class DenseNetDecoder(nn.Module):

    def __init__(self, layer_cfg, k, depth, conv_trans):
        super(DenseNetDecoder, self).__init__()
        self.depth = depth

        num_features = (((2 * k + layer_cfg[3] * k) // 2 + layer_cfg[2] * k) // 2 + layer_cfg[1] * k) // 2 + layer_cfg[0] * k

        if self.depth >= 5:
            self.block1 = DenseBlockDecoder(input_features=num_features, layers=layer_cfg[0], k=k)
        num_features -= layer_cfg[0] * k
        if self.depth >= 5:
            self.trans1 = TransitionDecoder(input_features=num_features, output_features=num_features*2, conv_trans=conv_trans)
        num_features = num_features * 2

        if self.depth >= 4:
            self.block2 = DenseBlockDecoder(input_features=num_features, layers=layer_cfg[1], k=k)
        num_features -= layer_cfg[1] * k
        if self.depth >= 4:
            self.trans2 = TransitionDecoder(input_features=num_features, output_features=num_features*2, conv_trans=conv_trans)
        num_features = num_features * 2
        
        if self.depth >= 3:
            self.block3 = DenseBlockDecoder(input_features=num_features, layers=layer_cfg[2], k=k)
        num_features -= layer_cfg[2] * k
        if self.depth >= 3:
            self.trans3 = TransitionDecoder(input_features=num_features, output_features=num_features*2, conv_trans=conv_trans)
        num_features = num_features * 2

        if self.depth >= 2:
            self.block4 = DenseBlockDecoder(input_features=num_features, layers=layer_cfg[3], k=k)
        num_features -= layer_cfg[3] * k

        if self.depth >= 2:
            if conv_trans:
                self.upsample = nn.ConvTranspose2d(in_channels=2*k, out_channels=2*k, kernel_size=3, stride=2, padding=1)
            else:
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            
        if conv_trans:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(num_features=2*k),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=2*k, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(num_features=2*k),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels=2*k, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False),
            )


        self.gate = nn.Sigmoid()
        
    
    def forward(self, x):

        if self.depth >= 5:
            x = self.block1(x)
            x = self.trans1(x)

        if self.depth >= 4:
            x = self.block2(x)
            x = self.trans2(x)

        if self.depth >= 3:
            x = self.block3(x)
            x = self.trans3(x)

        if self.depth >= 2:
            x = self.block4(x)
            x = self.upsample(x)

        x = self.conv1(x)

        x = self.gate(x)

        return x

class DenseNet(nn.Module):

    def __init__(self, layer_cfg, num_classes, dropout, k=32):
        super(DenseNet, self).__init__()

        self.encoder = DenseNetEncoder(layer_cfg=layer_cfg, k=k, depth=5)

        num_features = k // 4
        num_features += layer_cfg[3]*k
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=num_features, out_features=num_classes)
        )

        # The main difference lies in the weight initialization of convolutional and linear (fully connected) layers. The Kaiming He normal initialization is specifically tailored for layers followed by ReLU activations, aiming to maintain the variance of the activations throughout the layers. This can help prevent the vanishing or exploding gradients problem during training, especially in deep networks. In contrast, PyTorch's default initialization for linear layers uses a uniform distribution, which might not be as effective for deep networks with ReLU activations.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # same as default
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1) # same as default
                nn.init.constant_(m.bias, 0) # same as default
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0) # same as default
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class DenseLayer(nn.Module):

    def __init__(self, input_features, k):
        super(DenseLayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=4*k, kernel_size=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=4*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        return x
    
class DenseLayerDecoder(nn.Module):

    def __init__(self, out_features, k):
        super(DenseLayerDecoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=k),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=k, out_channels=4*k, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=4*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*k, out_channels=out_features, kernel_size=1, bias=False),
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        return x

class DenseBlockEncoder(nn.Module):

    def __init__(self, input_features, layers, k):
        super(DenseBlockEncoder, self).__init__()

        self.blocks = []

        for i in range(layers):

            #self.blocks.append(DenseLayer(input_features=input_features+i*k, k=k))

            layer = DenseLayer(input_features=input_features+i*k, k=k)

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():

            new_x = layer(x)

            x = torch.cat((x, new_x), 1)

        return x
    
class DenseBlockDecoder(nn.Module):

    def __init__(self, input_features, layers, k):
        super(DenseBlockDecoder, self).__init__()
        self.k = k

        self.blocks = []

        for i in range(layers):

            #self.blocks.append(DenseLayer(input_features=input_features+i*k, k=k))

            layer = DenseLayerDecoder(out_features=input_features-(i+1)*k, k=k)

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
                
                x, new_x = torch.split(x, [x.size(1)-self.k, self.k], dim=1)
                new_x = layer(new_x)
                x = torch.add(x, new_x)
                
        return x
    
class TransitionEncoder(nn.Module):

    def __init__(self, input_features, output_features):
        super(TransitionEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=1, bias=False),
        )

        self.avpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv1(x)

        x = self.avpool(x)

        return x
    
class TransitionDecoder(nn.Module):

    def __init__(self, input_features, output_features, conv_trans):
        super(TransitionDecoder, self).__init__()

        if conv_trans:
            self.upsample = nn.ConvTranspose2d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=output_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.upsample(x)
        
        x = self.conv1(x)

        return x
    
if __name__ == "__main__":

    
    input = torch.randn((5,3,224,224))
    print(input.shape)

    layer_cfg = get_configs("densenet121")

    encoder = DenseNetEncoder(layer_cfg=layer_cfg, k=32, depth=5)
    output = encoder(input)
    print(output.shape)

    decoder = DenseNetDecoder(layer_cfg=layer_cfg[::-1], k=32, depth=5, conv_trans=False)
    output = decoder(output)
    print(output.shape)

    autenc = DenseNetAutoEncoder(layer_cfg=layer_cfg, mirror=False, depth=5, conv_trans=False, freeze_enc=True)
    output = autenc(input)
    print(output.shape)

    model = DenseNet(layer_cfg, num_classes=2, dropout=0.0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Training Parameters', pytorch_total_params)

    #from torchsummary import summary
    #summary(encoder)
    #summary(decoder)
    #summary(autenc)
    #summary(model)



    