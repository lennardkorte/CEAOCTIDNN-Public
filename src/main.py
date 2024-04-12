import os
import json
import time
import torch
import argparse
import numpy as np

# Importing necessary modules and functions
from eval import Eval
from tqdm import tqdm
from utils import Utils, data_loader_sampling
from config import Config
from checkpoint import Checkpoint
from logger import Logger
from data_loaders import Dataloaders
from dataset_preparation import DatasetPreparation
from sklearn.utils.class_weight import compute_class_weight

# Function for training and evaluation
def train_and_eval(config:Config):    
    # Prepare dataset
    cust_data = DatasetPreparation(config)
    
    # Optionally show samples
    if config['show_samples']:
        data_loader_sampling(cust_data, config.save_path, config["dataset_no"], config['sample_no'])
    
    # Configure device for training
    device = Utils.config_torch_and_cuda(config)

    # Cross-validation loop
    only_x_cv = config['only_x_cv']
    for cv in range(config['num_cv']):
        if cv >= only_x_cv:
            break
        save_path_cv = config.save_path / ('cv_' + str(cv + 1))
        os.makedirs(save_path_cv, exist_ok=True)
        cv_done = True if any('metrics_test' in s for s in os.listdir(save_path_cv)) else False

        # Get indices for training and validation sets for current fold
        valid_ind_for_cv, train_ind_for_cv = cust_data.get_train_valid_ind(cv)
        # Setup data loaders for training
        Dataloaders.setup_data_loaders_training(train_ind_for_cv, train_ind_for_cv[::config['num_cv']], valid_ind_for_cv, cust_data, config)

        # Compute class weights for balancing
        cv_labels = cust_data.label_data[train_ind_for_cv]
        class_weights_tensor = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(cv_labels), y=cv_labels), dtype=torch.float)
        
        if not cv_done:
            # Initialize checkpoint and logger
            checkpoint = Checkpoint('checkpoint_last', save_path_cv, device, config, cv + 1)
            Logger.init(cv+1, checkpoint, config)
            
            epoch_impr_and_no_overfitting = checkpoint.epoch_impr_and_no_overfitting
            
            # Training loop
            for epoch in tqdm(range(checkpoint.epoch, config['epochs']+1), total=config['epochs']-checkpoint.epoch+1, desc=f'Training epochs (CV {cv+1})'):
                if epoch - epoch_impr_and_no_overfitting > config['early_stop_patience'] and config['early_stop_patience']:
                    break
                
                start_it_epoch = time.time()
                Utils.train_one_epoch(checkpoint.model, device, checkpoint.scaler, checkpoint.optimizer, config, class_weights_tensor)
                Logger.add({'epoch': epoch, 'epoch_training_duration': time.time() - start_it_epoch})

                checkpoint.scheduler.step()
                
                # Evaluate on validation set
                eval_valid = Eval(Dataloaders.validation, device, checkpoint.model, config, save_path_cv, cv + 1, class_weights=class_weights_tensor)
                Logger.add(eval_valid.metrics, 'val_set')

                # Optionally calculate training error
                if config['calc_train_error']:
                    eval_train = Eval(Dataloaders.training_eval, device, checkpoint.model, config, save_path_cv, cv+1, class_weights=class_weights_tensor)
                    Logger.add(eval_train.metrics, 'train_set_peak')
                
                # Check for overfitting
                if config['calc_train_error']:
                    if config["auto_encoder"]:
                        overfitting = round(eval_valid.metrics['mean_loss'], config['early_stop_accuracy']) < round(eval_train.metrics['mean_loss'], config['early_stop_accuracy'])
                    else:
                        overfitting = round(eval_valid.metrics['bal_acc'], config['early_stop_accuracy']) >= round(eval_train.metrics['bal_acc'], config['early_stop_accuracy'])
                else:
                    overfitting

                # Check for improvement
                if epoch > 1:
                    if config["auto_encoder"]:
                        improvement_identified = round(eval_valid.metrics['mean_loss'], config['early_stop_accuracy']) < round(checkpoint.eval_valid_best_metrics['mean_loss'], config['early_stop_accuracy'])
                    else:
                        improvement_identified = round(eval_valid.metrics['bal_acc'], config['early_stop_accuracy']) > round(checkpoint.eval_valid_best_metrics['bal_acc'], config['early_stop_accuracy'])
                else:
                    improvement_identified = True

                # Save best model
                if (improvement_identified and not overfitting) or epoch == 1:
                    checkpoint.save_checkpoint('checkpoint_best', epoch, epoch, eval_valid.metrics, save_path_cv, config)
                    Checkpoint.delete_checkpoint('checkpoint_best', epoch_impr_and_no_overfitting, save_path_cv)
                    epoch_impr_and_no_overfitting = epoch
                Logger.add({'best_epoch': epoch_impr_and_no_overfitting})
 
                # Optionally evaluate on test set
                if config['calc_and_peak_test_error']:
                    if Dataloaders.test is None: Dataloaders.setup_data_loader_testset(cust_data, config)
                    eval_test = Eval(Dataloaders.test, device, checkpoint.model, config, save_path_cv, cv + 1, class_weights=class_weights_tensor)
                    Logger.add(eval_test.metrics, 'test_set')

                # Save current model
                checkpoint.save_checkpoint('checkpoint_last', epoch, epoch_impr_and_no_overfitting, eval_valid.metrics, save_path_cv, config)
                if epoch > 1: Checkpoint.delete_checkpoint('checkpoint_last', epoch - 1, save_path_cv)

                Logger.push(save_path_cv / 'log_train.csv')

        # Aggregate test metrics for both last and best checkpoints
        metrics_test = {}
        for checkpoint_name in ['checkpoint_last', 'checkpoint_best']:
            checkpoint = Checkpoint(checkpoint_name, save_path_cv, device, config, cv+1)
            if Dataloaders.test is None: Dataloaders.setup_data_loader_testset(cust_data, config)
            eval_test = Eval(Dataloaders.test, device, checkpoint.model, config, save_path_cv, cv+1, checkpoint_name=checkpoint_name, class_weights=class_weights_tensor)
            selected_keys = ['mean_loss', 'bal_acc', 'accuracy', 'sens', 'spec', 'f1', 'mcc', 'prec']
            checkpoint_metrics = {k: eval_test.metrics[k] for k in selected_keys if k in eval_test.metrics}
            metrics_test[checkpoint_name] = checkpoint_metrics
            Logger.add(checkpoint_metrics, checkpoint_name)
        with open(save_path_cv / 'metrics_test.json', 'w') as f: json.dump(metrics_test, f, indent=4)
        Logger.push(save_path_cv / 'metrics_test.csv')

    # Aggregate and compute average test metrics across cross-validation folds
    Logger.init(0, checkpoint, config)
    accumulated_metrics = {'checkpoint_last': {}, 'checkpoint_best': {}}
    for cv in range(config['num_cv']):
        if cv >= only_x_cv:
            break
        with open(config.save_path / f'cv_{cv + 1}' / 'metrics_test.json', 'r') as f:
            metrics_test = json.load(f)
            for checkpoint_name in ['checkpoint_last', 'checkpoint_best']:
                for metric, value in metrics_test[checkpoint_name].items():
                    if metric not in accumulated_metrics[checkpoint_name]:
                        accumulated_metrics[checkpoint_name][metric] = []
                    accumulated_metrics[checkpoint_name][metric].append(value)
            
    average_metrics = {checkpoint: {} for checkpoint in ['checkpoint_last', 'checkpoint_best']}
    for checkpoint_name in ['checkpoint_last', 'checkpoint_best']:
        for metric, values in accumulated_metrics[checkpoint_name].items():
            average_metrics[checkpoint_name][metric] = sum(values) / len(values)
        Logger.add(average_metrics, checkpoint_name + '_avg')
    Logger.push(config.save_path / 'metrics_test_avg.csv')
    with open(config.save_path / 'metrics_test_avg.json', 'w') as f: json.dump(average_metrics, f, indent=4)


if __name__ == '__main__':
    # Parse command line arguments
    args = argparse.ArgumentParser(description='CEAOCTIDNN')
    args.add_argument('-cfg', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-gpu', '--gpus', default='0', type=str, help='indices of GPUs to enable (default: all)') # TODO
    args.add_argument('-wb', '--wandb', default=None, type=str, help='Wandb API key (default: None)')
    args.add_argument('-ntt', '--no_trainandtest', dest='trainandtest', action='store_false', help='Deactivation of Training and Testing (default: Activated)')
    args.add_argument('-smp', '--show_samples', dest='show_samples', action='store_true', help='Activate creation of Sample from Data Augmentation (default: Deactivated)')
    args.add_argument('-ycf', '--overwrite_configurations', dest='overwrite_configurations', action='store_true', help='Overwrite Configurations, if config file in this directory already exists. (default: False)')
    args.add_argument('-lr', '--learning_rate', default=3e-6, type=float, help='')
    args.add_argument('-wd', '--weight_decay', default=0.001, type=float, help='')
    args.add_argument('-mo', '--momentum', default=0.9, type=float, help='')
    args.add_argument('-bs', '--batch_size', default=128, type=int, help='')
    args.add_argument('-nm', '--name', default='run_0', type=str, help='')
    args.add_argument('-gr', '--group', default='group_0', type=str, help='')
    
    config = Config(args)
    
    train_and_eval(config)
