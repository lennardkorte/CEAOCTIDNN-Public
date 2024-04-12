import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage
#from image_transforms import CircularMask
from scipy.stats import linregress
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef


class Eval():
    def __init__(self, dataloader, device, model, config, save_path_cv, cv, checkpoint_name=None, class_weights=None):
        model.eval()

        self.metrics = {}
        tests = [False]
        if config['MCdropout_test'] and not config['auto_encoder'] and checkpoint_name is not None:
            tests.append(True)
        for mc_dropout_test in tests:
            if mc_dropout_test:
                loss_linear_all_list = []
                mc_iterator = tqdm(range(config['mc_iterations']), total=config['mc_iterations'], desc='MC Confidence Estimation', leave=False)
            else:
                mc_iterator = range(1)
            for _ in mc_iterator:
                for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'MC Evaluation' if mc_dropout_test else 'Evaluation', leave=False):
                    inputs = inputs.to(device)
                    labels_long = labels.type(torch.LongTensor).to(device)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(False):
                        with torch.cuda.amp.autocast():
                            
                            if mc_dropout_test:
                                for m in model.modules():
                                    if m.__class__.__name__.startswith('Dropout'):
                                        m.train()

                            outputs = model(inputs)
                            
                            if config["auto_encoder"]:
                                '''
                                # Compute Loss without log # TODO: mask for training as well
                                inputs = CircularMask(0.9)(inputs)
                                inputs = CircularMask(0.17, True)(inputs)
                                outputs = CircularMask(0.9)(outputs)
                                outputs = CircularMask(0.17, True)(outputs)'''

                                me_loss_function = nn.L1Loss(reduction='none') # Mean Error
                                loss_linear_elementwise = me_loss_function(outputs, inputs)
                                loss_linear_each = torch.mean(loss_linear_elementwise, dim=[1,2,3])

                                if not mc_dropout_test:
                                    mse_loss_function = nn.MSELoss(reduction='none')
                                    loss_elementwise = mse_loss_function(outputs, inputs)
                                    loss_each = torch.mean(loss_elementwise, dim=[1,2,3]) # loss_each is loss for each image in batch (8 floats for batch size 8)
                            else:
                                # cross-entropy loss without log
                                softmax_function = nn.Softmax(dim=1)
                                outputs_softmax = softmax_function(outputs)
                                nllloss_function = nn.NLLLoss(reduction='none')
                                loss_linear_each = nllloss_function(outputs_softmax, labels_long)

                                if not mc_dropout_test:
                                    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='none') # no weights here
                                    loss_each = ce_loss(outputs, labels_long)
                        
                    if i == 0:
                        loss_linear_all_tensor = loss_linear_each
                        if not mc_dropout_test:
                            loss_all_tensor = loss_each
                            targets_all_tensor = labels_long
                            predictions_all_tensor = outputs
                            inputs_all_tensor = inputs
                            
                    else:
                        loss_linear_all_tensor = torch.cat([loss_linear_all_tensor, loss_linear_each], 0)
                        if not mc_dropout_test:
                            loss_all_tensor = torch.cat([loss_all_tensor, loss_each], 0)
                            targets_all_tensor = torch.cat([targets_all_tensor, labels_long], 0)
                            predictions_all_tensor = torch.cat([predictions_all_tensor, outputs], 0)
                            inputs_all_tensor = torch.cat([inputs_all_tensor, inputs], 0)
                            
                # Transform to numpy arrays
                loss_linear_all = loss_linear_all_tensor.cpu().numpy()
                
                #mean_loss_linear = np.mean(loss_linear_all)
                if mc_dropout_test:
                    loss_linear_all_list.append(loss_linear_all)
                else:
                    loss_all = loss_all_tensor.cpu().numpy()
                    mean_loss = np.mean(loss_all)
                    self.metrics['mean_loss'] = float(mean_loss)
                    targets_np = targets_all_tensor.cpu().numpy()
                    if not config['auto_encoder']:
                        predictions_np = predictions_all_tensor.cpu().numpy()

            if not config["auto_encoder"] and not mc_dropout_test:
                self.metrics.update(self.calc_metrics(predictions_np, targets_all_tensor))
                
            if checkpoint_name is not None: # only given when testing best and last checkpoint
                if not config["auto_encoder"]:
                    predicted_labels = np.argmax(predictions_np, axis=1)
                    if mc_dropout_test:                      
                        losses = np.array(loss_linear_all_list)
                        variances = np.var(losses, axis=0)
                        self.uncertainty_confusion_matrix(variances, predicted_labels, targets_np, save_path_cv / (checkpoint_name + '_avg_var.json'))
                        self.loss_distribution_analysis(variances, loss_linear_all, predicted_labels, targets_np, save_path_cv, checkpoint_name + '_' + 'variance')
                    else:
                        predictions_file_name = save_path_cv / (checkpoint_name + '_predloss_pairs.txt')
                        with open(predictions_file_name, 'w') as file:
                            for int_class, float_loss in zip(predicted_labels,loss_linear_all):
                                file.write(f"{int_class},{float_loss}\n")

                else:
                    self.save_auto_encoder_sample(inputs_all_tensor, predictions_all_tensor, save_path_cv)
                    if config['compare_classifier_predictions']:
                        save_path_classifier = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
                        predictions_file_name = save_path_classifier / (checkpoint_name + '_predloss_pairs.txt')
                        classifier_predloss_pairs = []
                        with open(predictions_file_name, 'r') as file:
                            for line in file:
                                parts = line.strip().split(',')
                                classifier_predloss_pairs.append((int(parts[0]), float(parts[1])))
                        predicted_labels, classifier_losses = tuple(map(list, zip(*classifier_predloss_pairs)))
                        predicted_labels = np.array(predicted_labels)
                        classifier_losses = np.array(classifier_losses)
                        
                        self.uncertainty_confusion_matrix(loss_linear_all, predicted_labels, targets_np, save_path_cv / (checkpoint_name + '_avg_autenc_loss.json'))
                        self.loss_distribution_analysis(loss_linear_all, classifier_losses, predicted_labels, targets_np, save_path_cv, checkpoint_name + '_' + 'CEB_loss')


    @staticmethod
    def uncertainty_confusion_matrix(uncertainty_indicator, predicted_labels, targets, path):
        if not ((len(uncertainty_indicator) == len(predicted_labels)) and (len(predicted_labels) == len(targets))):
            assert AssertionError("The classifier was tested on a different number of images!")
        df = pd.DataFrame({'true_labels': targets, 'predicted_labels': predicted_labels, 'variance': uncertainty_indicator})
        average_variances = df.groupby(['true_labels', 'predicted_labels'])['variance'].mean().unstack(fill_value=0)
        average_variances_dict = average_variances.to_dict()
        with open(path, 'w') as f:
            json.dump(average_variances_dict, f, indent=4)

    @staticmethod
    def loss_distribution_analysis(ue_values, classifier_losses, predicted_labels, targets_np, save_path_cv, store_name):
            # Apply Threshold
            classifier_losses = np.array(classifier_losses)
            #mask1 = np.array([elem > -1 for elem in classifier_losses])
            #mask2 = np.array([elem > -10.0 for elem in ue_values])
            mask1 = np.array([True for elem in classifier_losses])
            mask2 = np.array([True for elem in ue_values])
            classifier_losses_threshold = classifier_losses[mask1 & mask2]
            ue_values_threshold = ue_values[mask1 & mask2]

            # Create plot Loss Pair Distribution of Classifier vs Autoencoder
            colors = np.array([])
            for t, p in zip(targets_np, predicted_labels): #colors = np.array([('green' if t == p else 'red') for t, p in zip(targets_all_tensor, classifier_predictions)])
                color = 'green' if t == p else 'red' # TODO: colors not right
                colors = np.append(colors, color)
            colors_threshold = colors[mask1 & mask2]

            x = ue_values_threshold
            y = classifier_losses_threshold
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line = slope * x + intercept
            pearson_coefficient = r_value
            print("slope: ", slope)
            print("pearson_coefficient: ", pearson_coefficient)

            plt.clf()
            plt.figure(figsize=(10, 6))
            plt.scatter(ue_values_threshold, classifier_losses_threshold, alpha=0.5, c=colors_threshold)
            plt.plot(x, line, color='blue', label='Linear Regression Line')
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid(True)
            plt.ylim(-1, 0)
            plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.1)
            plt.savefig(save_path_cv / (store_name + '_distr_plot.png'))
            plt.close()

            # Create plot with Proportion of Samples with Lowest MAE
            ue_values_asc, classifier_losses_asc = zip(*sorted(zip(ue_values_threshold, classifier_losses_threshold), key=lambda x: x[0]))
            ue_values_asc = list(ue_values_asc)
            classifier_losses_asc = list(classifier_losses_asc)
            proportions = np.linspace(0.01, 1, 100)  # 100 proportions from 1% to 100%
            average_losses = [] # average_losses contains the average classifier loss for each proportion
            for proportion in proportions:
                n_samples = int(proportion * len(classifier_losses_asc))
                selected_losses = classifier_losses_asc[:n_samples]
                average_loss = np.mean(selected_losses)
                average_losses.append(average_loss)
            plt.clf()
            plt.figure(figsize=(10, 6))
            plt.plot(proportions, average_losses)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15)
            plt.grid(True)
            plt.ylim(-1, 0)
            plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.1)
            plt.savefig(save_path_cv / (store_name + '_risk_cov_curve.png'))
            plt.close()

            # Initialize lists to store results
            thresholds = np.arange(0.01, 1.01, 0.01)
            balanced_accuracies = []

            # Calculate balanced accuracy for different thresholds
            for threshold in thresholds:
                mask = ue_values < threshold
                if np.sum(mask) > 0:  # Ensure there are samples below the threshold
                    predictions_threshold = predicted_labels[mask]
                    targets_threshold = targets_np[mask]
                    balanced_accuracy = balanced_accuracy_score(targets_threshold, predictions_threshold)
                else:
                    balanced_accuracy = None
                balanced_accuracies.append(balanced_accuracy)

            plt.clf()
            plt.figure(figsize=(10, 6))
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, balanced_accuracies, marker='o', linestyle='-', color='blue')
            plt.xlabel('Threshold', fontsize=20)
            plt.ylabel('Balanced Accuracy', fontsize=20)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15)
            plt.grid(True)
            plt.ylim(0, 1)
            plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.12)
            plt.savefig(save_path_cv / (store_name + '_bacc_thresh.png'))
            plt.close()

    @staticmethod
    def save_auto_encoder_sample(inputs_all_tensor, predictions_all_tensor, save_path_cv):
        num_samples = 10
        for i, (input_tensor, prediction_tensor) in enumerate(zip(inputs_all_tensor, predictions_all_tensor)):
            if not i % int(len(predictions_all_tensor) / num_samples):
                # Extract the slice (single channel image) from the tensor
                image_slice_in = input_tensor[0, :, :]
                image_slice_pred = prediction_tensor[0, :, :]

                '''
                # with filters:
                image_slice_in_filtered = input_tensor[0, :, :]
                image_slice_in_filtered = T.GaussianBlur(kernel_size=(5,5), sigma=(2,2))(image_slice_in_filtered.unsqueeze(0)).squeeze(0)
                from da_techniques import CircularMask
                image_slice_in_filtered = CircularMask(radius=0.1)(image_slice_in_filtered)
'''
                
                images_max = max(image_slice_in.max(), image_slice_pred.max())
                images_min = min(image_slice_in.min(), image_slice_pred.min())
                image_in = (255 * (image_slice_in - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                # image_in_filtered = (255 * (image_slice_in_filtered - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                image_out = (255 * (image_slice_pred - images_min) / (images_max - images_min)).clamp(0, 255).byte()

                to_pil = ToPILImage()
                image_in_pil = to_pil(image_in)
                # image_in_filtered_pil = to_pil(image_in_filtered)
                image_out_pil = to_pil(image_out)
                img_dir = save_path_cv / "example_images/"
                os.makedirs(img_dir, exist_ok = True)
                image_in_pil.save(img_dir / f"{i}_input.png", "PNG")
                # image_in_filtered_pil.save(img_dir / f"{i}_input_blur.png", "PNG")
                image_out_pil.save(img_dir / f"{i}_output.png", "PNG")

                # Print absolute difference of input and output
                abs_diff = torch.abs(torch.subtract(image_slice_in, image_slice_pred))
                abs_diff = (255 * (abs_diff - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                #Alternative min and max values when normalizing for range 0 to 255
                #abs_diff = (255 * (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min())).clamp(0, 255).byte()
                abs_diff_image = abs_diff.cpu().numpy()
                abs_diff_image_np = Image.fromarray(np.uint8(abs_diff_image), mode='L')
                abs_diff_image_np.save(img_dir / f"{i}_absdiff.png")
                '''
                segments = skimage.segmentation.slic(abs_diff_image, n_segments=8, compactness=0.03, channel_axis=None)
                segmentation_overlay = skimage.color.label2rgb(segments, image=abs_diff_image, kind='overlay')
                segmentation_image = Image.fromarray(np.uint8(segmentation_overlay * 255))
                segmentation_image.save(img_dir / f"{i}_segmentation.png")
                '''
    
    @staticmethod
    def calc_metrics(predictions_np, targets):
        
        metrics = {
            "bal_acc": 0.0,
            "accuracy": 0.0,
            "sens": 0.0,
            "spec": 0.0,
            "f1": 0.0,
            "mcc": -1.0,
            "prec": 0.0
        }

        targets_np = targets.cpu().numpy()
        predicted_classes = np.argmax(predictions_np, axis=1)

        try:
            metrics["accuracy"] = accuracy_score(targets_np, predicted_classes)
            metrics["bal_acc"] = balanced_accuracy_score(targets_np, predicted_classes)
            metrics["prec"] = precision_score(targets_np, predicted_classes, average='weighted', zero_division=0)
            metrics["sens"] = recall_score(targets_np, predicted_classes, average='weighted', zero_division=0)
            metrics["f1"] = f1_score(targets_np, predicted_classes, average='weighted', zero_division=0)
            metrics["mcc"] = matthews_corrcoef(targets_np, predicted_classes)

            # Computing specificity requires a confusion matrix
            cm = confusion_matrix(targets_np, predicted_classes)
            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tp = cm[1, 1]

            # Calculate specificity
            metrics["spec"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception as e:
            print(f"Error computing metrics: {e}")

        return metrics
        
        