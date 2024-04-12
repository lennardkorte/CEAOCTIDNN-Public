import os
import torch
import random
import json

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from pathlib import Path
from collections import OrderedDict

from logger import Logger  # Importing Logger class from logger module
from dataset import OCT_Dataset  # Importing OCT_Dataset class from dataset module
from data_loaders import Dataloaders  # Importing Dataloaders class from data_loaders module

import torch
import os
from torchvision.utils import save_image

from pathlib import Path

class Utils():
    ''' 
    Utility class providing various helper functions.
    '''

    @staticmethod
    def config_torch_and_cuda(config):
        ''' 
        Configure PyTorch and CUDA based on provided configuration.
        '''
        # Setting CUDA_VISIBLE_DEVICES environment variable to specify GPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' #config['gpus']

        print("Indices of devices to use:", os.environ["CUDA_VISIBLE_DEVICES"])

        # Set location where torch stores its models
        os.environ['TORCH_HOME'] = './data/torch_pretrained_models'

        torch.backends.cudnn.enabled = config['use_cuda']

        # Use deterministic training if specified
        if config['deterministic_training']:
            seed = 18
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if config['deterministic_batching']:
                np.random.seed(seed)
                random.seed(seed)

        try:
            # Check if GPU is available
            assert torch.cuda.is_available(), "No GPU available"
            if config['gpu'] == 1:
                assert 2 <= torch.cuda.device_count(), "Second GPU not available"
            print("Chosen GPU is available")
            print(torch.cuda.get_device_name(config['gpu']))

        except AssertionError as error:
            # Handle the assertion error if no GPU is available
            print(f"Assertion Error: {error}")
            raise SystemExit("Program terminated due to lack of GPU.")

        return torch.device(config['gpu'])

    @staticmethod
    def train_one_epoch(model, device, scaler, optimizer, config, class_weights):
        '''
        Train the model for one epoch.

        Args:
        - model: PyTorch model to be trained
        - device: Device to perform training (CPU or GPU)
        - scaler: PyTorch scaler for mixed precision training
        - optimizer: PyTorch optimizer for updating model parameters
        - config: Configuration dictionary
        - class_weights: Weights for different classes used in loss calculation
        '''
        # Define loss function based on configuration
        if config['auto_encoder']:
            loss_function = nn.MSELoss()
        else:
            loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))

        model.train()  # Set model to training mode

        learning_rate_sum = 0
        loss_sum = 0
        # Iterate over training data batches
        for j, (inputs, labels) in tqdm(enumerate(Dataloaders.training), total=len(Dataloaders.training), desc='Training batches', leave=False):

            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)

            optimizer.zero_grad()  # Clear gradients from previous iteration

            with torch.set_grad_enabled(True):

                # Runs the forward pass under autocast for mixed precision training
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)  # Forward pass

                    if config["auto_encoder"]:
                        # Compute loss for autoencoder
                        loss_all = loss_function(outputs, inputs)
                    else:
                        loss_all = loss_function(outputs, labels)  # Compute loss

                # Scale loss and call backward() to create scaled gradients
                scaler.scale(loss_all).backward()

                # Unscale gradients and call or skip optimizer.step()
                scaler.step(optimizer)

                # Update the scale for next iteration
                scaler.update()

                learning_rate_sum += optimizer.param_groups[0]['lr']  # Accumulate learning rates
                loss_sum += loss_all  # Accumulate loss

        # Log mean learning rate for the epoch
        Logger.add({"mean_learning_rate": learning_rate_sum / (j + 1)}, "train_set")

    @staticmethod
    def read_json(file_name):
        '''
        Read JSON data from a file.

        Args:
        - file_name: Path to the JSON file

        Returns:
        - JSON data
        '''
        file_name = Path(file_name)
        with file_name.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)

    @staticmethod
    def write_json(content, file_name):
        '''
        Write JSON data to a file.

        Args:
        - content: JSON data to be written
        - file_name: Path to the JSON file
        '''
        file_name = Path(file_name)
        with file_name.open('wt') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)

def data_loader_sampling(cust_data, path, dataset_no, sample_no):
    '''
    Sample and preprocess data from a custom dataset.

    Args:
    - cust_data: Custom dataset containing images and labels
    - path: Path to save sampled images
    - dataset_no: Identifier for the dataset
    - sample_no: Number of samples to be extracted
    '''
    os.makedirs(path / 'sample_images', exist_ok=True)  # Create directory to save sampled images
    sample_ind = random.sample(range(len(cust_data.label_data)), sample_no)  # Sample indices

    # Initialize dataset for preprocessing and augmentation
    dataset_prepro = OCT_Dataset(sample_ind, cust_data.label_data, cust_data.all_files_paths, False, False, dataset_no)
    dataset_prepro_and_aug = OCT_Dataset(sample_ind, cust_data.label_data, cust_data.all_files_paths, True, False, dataset_no)

    # Iterate over samples
    for i in range(sample_no):
        # Get preprocessed image and label
        image_prepro, image_prepro_label = dataset_prepro[i]
        image_prepro_and_aug, image_prepro_and_aug_label = dataset_prepro_and_aug[i]

        # Save preprocessed and augmented images
        image_prepro_rescaled = image_prepro
        image_prepro_and_aug_rescaled = image_prepro_and_aug

        save_image(image_prepro_rescaled, path / f'sample_images/prepro_{i+1}.png')
        save_image(image_prepro_and_aug_rescaled, path / f'sample_images/prepro_and_aug_{i+1}.png')
