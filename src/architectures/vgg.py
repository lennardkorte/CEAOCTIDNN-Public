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

    if arch == 'vgg11':
        layer_cfg = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        layer_cfg = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        layer_cfg = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        layer_cfg = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")
    
    return layer_cfg

class VGGAutoEncoder(nn.Module):

    def __init__(self, layer_cfg, mirror, freeze_enc, depth):

        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = VGGEncoder(layer_cfg=layer_cfg, depth=depth, enable_bn=True)
        if freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if mirror:
            self.decoder = VGGDecoder(layer_cfg=layer_cfg[::-1], depth=depth, enable_bn=True)
        else:
            layer_cfg, bottleneck = resnet.get_configs("resnet50")
            self.decoder = resnet.ResNetDecoder(layer_cfg=layer_cfg[::-1], bottleneck=bottleneck, conv_trans=False, depth=depth, in_channels=[512, 512, 256, 128, 64])
        
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class VGG(nn.Module):

    def __init__(self, layer_cfg, num_classes, dropout, img_size, enable_bn=False):
        super(VGG, self).__init__()

        self.encoder = VGGEncoder(layer_cfg=layer_cfg, depth=5, enable_bn=enable_bn)

        self.img_size = img_size / 32

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.img_size*self.img_size*512), out_features=4096),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # init like in torchvision
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        x = self.encoder(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class VGGEncoder(nn.Module):

    def __init__(self, layer_cfg, depth, enable_bn=True):
        self.depth = depth

        super(VGGEncoder, self).__init__()

        if len(layer_cfg) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = EncoderBlock(input_dim=3,   output_dim=64,  hidden_dim=64,  layers=layer_cfg[0], enable_bn=enable_bn)
        if self.depth >= 2:
            self.conv2 = EncoderBlock(input_dim=64,  output_dim=128, hidden_dim=128, layers=layer_cfg[1], enable_bn=enable_bn)
        if self.depth >= 3:
            self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=layer_cfg[2], enable_bn=enable_bn)
        if self.depth >= 4:
            self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=layer_cfg[3], enable_bn=enable_bn)
        if self.depth >= 5:
            self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=layer_cfg[4], enable_bn=enable_bn)
    
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

class VGGDecoder(nn.Module):

    def __init__(self, layer_cfg, depth, enable_bn=True):
        super(VGGDecoder, self).__init__()
        self.depth = depth

        if len(layer_cfg) != 5:

            raise ValueError("There should be 5 stage in VGG")

        if self.depth >= 5:
            self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=layer_cfg[0], enable_bn=enable_bn)
        if self.depth >= 4:
            self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=layer_cfg[1], enable_bn=enable_bn)
        if self.depth >= 3:
            self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=layer_cfg[2], enable_bn=enable_bn)
        if self.depth >= 2:
            self.conv4 = DecoderBlock(input_dim=128, output_dim=64,  hidden_dim=128, layers=layer_cfg[3], enable_bn=enable_bn)
        
        self.conv5 = DecoderBlock(input_dim=64,  output_dim=3,   hidden_dim=64,  layers=layer_cfg[4], enable_bn=enable_bn)
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

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d EncoderLayer' % i, layer)
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d DecoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x):

        return self.layer(x)

class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
    
    def forward(self, x):

        return self.layer(x)

if __name__ == "__main__":

    input = torch.randn((5,3,224,224))
    print(input.shape)

    layer_cfg = get_configs('vgg19')

    encoder = VGGEncoder(layer_cfg, depth=5)
    output = encoder(input)
    print(output.shape)

    decoder = VGGDecoder(layer_cfg, depth=5)
    output = decoder(output)
    print(output.shape)

    autenc = VGGAutoEncoder(layer_cfg=layer_cfg, mirror=False, freeze_enc=True, depth=5)
    output = autenc(input)
    print(output.shape)

    model = VGG(layer_cfg, num_classes=2, dropout=0.0, img_size=input.shape[-1])
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Training Parameters', pytorch_total_params)

    #from torchsummary import summary
    #summary(encoder)
    #summary(decoder)
    #summary(autenc)
    #summary(model)



    