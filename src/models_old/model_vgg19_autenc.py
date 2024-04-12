
# Inspired by: https://github.com/Horizon2333/imagenet-autoencoder/blob/536633ec1c0e9afe2dd91ce74b56e6e13479b6bd/models/resnet.py
# (This file manually builds Resnets with different configurations and their respective autoencoders)

import torch.nn as nn
import torch
from torchvision import models
from pathlib import Path
from glob import glob

'''
        from models.model_unet1 import UNetClassifier1, load_unet1_with_classifier_weights
        from models.model_unet2 import UNetClassifier2, load_unet2_with_classifier_weights
        'VGG19': VGG19,
        'VGG19AutEnc': create_autoenc_vgg19,
        'UNetClassifier1': UNetClassifier1,
        'load_unet1_with_classifier_weights': load_unet1_with_classifier_weights,
        'UNetClassifier2': UNetClassifier2,
        'load_unet2_with_classifier_weights': load_unet2_with_classifier_weights,'''


class VGG19(nn.Module):
    def __init__(self, config, cv):
        super(VGG19, self).__init__()
        self.output_size = config['num_classes']
        weights = None
        if config['pretrained']: weights = models.VGG19_Weights.IMAGENET1K_V1
        self.net = models.vgg19(weights=weights)

        # Adding dropout to convolutional layers
        feats_list = list(self.net.features)
        new_feats_list = []
        for feat in feats_list:
            new_feats_list.append(feat)
            if isinstance(feat, nn.Conv2d):
                new_feats_list.append(nn.Dropout(p=0.5, inplace=True))
        self.net.features = nn.Sequential(*new_feats_list)

        # Modify the classifier - Adding Dropout
        self.net.classifier = nn.Sequential(
            *list(self.net.classifier.children())[:-1],  # Keep all layers except the last one
            nn.Dropout(config['dropout']),  # Add Dropout here
            nn.Linear(4096, self.output_size)  # The final Linear layer with desired output size
        )

    def forward(self, x):
        return self.net(x)

class ResNetDecoder(nn.Module):

    # use inverted layer_cfg as argument to create decoder, i.e.g: layer_cfg[::-1]
    def __init__(self, layer_cfg, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(layer_cfg) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=layer_cfg[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=layer_cfg[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=layer_cfg[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=layer_cfg[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=layer_cfg[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=layer_cfg[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=layer_cfg[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=layer_cfg[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
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

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
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
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
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

# Create the custom autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_autoenc_vgg19(config, cv):

    # Load the pretrained ResNet18 model from a ".pt" file
    save_path_ae_cv = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
    for path in glob(str(save_path_ae_cv / '*.pt')):
        if "checkpoint_best" in path:
            checkpoint_path = path
    
    resnet = VGG19(config, cv)

    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['model_state_dict'])

    # Define the encoder using the first layers of the model
    encoder = nn.Sequential(*list(resnet.net.children())[:-2])

    # Set the encoder layers to be non-trainable
    for param in encoder.parameters():
        param.requires_grad = False

    arch, bottleneck = [2, 2, 2, 2], False
    decoder = ResNetDecoder(arch[::-1], bottleneck=bottleneck)

    autoencoder = Autoencoder(encoder, decoder)

    return autoencoder
