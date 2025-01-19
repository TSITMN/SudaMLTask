import random
import numpy as np
import torch
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import pandas as pd

def set_seed(seed):
    """ Set random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=1e-5, lr_max=1e-4, warmup_epoch=10):
    """
    Update the optimizer's learning rate with Warmup + Cosine Annealing.

    Args:
        optimizer (Optimizer): Optimizer.
        current_epoch (int): Current epoch.
        max_epoch (int): Total training epochs.
        lr_min (float): Minimum learning rate.
        lr_max (float): Maximum learning rate.
        warmup_epoch (int): Number of warmup epochs.
    """
    if current_epoch < warmup_epoch:
        # Warmup 
        lr = lr_max * (current_epoch + 1 ) / (warmup_epoch + 1)
    else:
        # Cosine annealing
        progress = (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch)
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * progress)) / 2

    print("lr = " , lr)

    # Update the optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
def calculate_model_metrics(model, data_loader, scaler=None, device='cuda'):
    """ Evaluate the LSTM model on the given data loader with MSE and MAE. """
    model.eval()
    mse = 0
    mae = 0
    inputs = []
    predictions = []
    targets = []
    mse_list = []
    mae_list =[]
    # mean = scaler.mean_[-1]
    # std = np.sqrt(scaler.var_[-1])
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Testing", unit="batch"):
            target = target[:, :, -1]
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            inputs.append(data[:, :, -1].cpu().numpy())
            for i in range(len(target)):
                # tgt = inverse_transform(target[i].cpu().numpy(), mean, std)
                # out = inverse_transform(output[i].cpu().numpy(), mean, std)
                mse_list.append(mean_squared_error(target[i].cpu().numpy(), output[i].cpu().numpy()))
                mae_list.append(mean_absolute_error(target[i].cpu().numpy(), output[i].cpu().numpy()))
                
    mae = mean_absolute_error(np.concatenate(targets), np.concatenate(predictions))
    mse = mean_squared_error(np.concatenate(targets), np.concatenate(predictions))
   
    return mse, mae, np.concatenate(predictions), np.concatenate(targets), np.concatenate(inputs), mse_list, mae_list

def inverse_transform(x, mean, std):
    """ Inverse transform the scaled data. """
    return x * std + mean

def plot_predictions(inputs, predictions, targets, input_seq_len=96, output_seq_len=96,
                     mse_list=None, mae_list=None, save_path='./results/', model_suffix='', scaler=None):
    """ Plot the predictions and true values.
    
    Args:
        inputs (numpy array): Input sequences.
        predictions (numpy array): Predicted sequences.
        targets (numpy array): True sequences.
        input_seq_len (int): Length of input sequences.
        output_seq_len (int): Length of output sequences.
        mse_list (list): List of MSE values for each sample.
        mae_list (list): List of MAE values for each sample.
        save_path (str): Path to save the plot.
        scaler (StandardScaler): Scaler object to inverse transform the data.
    """
    plot_save_dir = os.path.join(save_path, model_suffix)
    os.makedirs(plot_save_dir, exist_ok=True)
    
    # inverse transform
    if scaler is not None:
        print(scaler.mean_[-1], scaler.var_[-1])
        inputs = inverse_transform(inputs, scaler.mean_[-1], np.sqrt(scaler.var_[-1]))
        predictions = inverse_transform(predictions, scaler.mean_[-1], np.sqrt(scaler.var_[-1]))
        targets = inverse_transform(targets, scaler.mean_[-1], np.sqrt(scaler.var_[-1]))
        # print(inputs.shape, predictions.shape, targets.shape)
    
    if mse_list is not None:
        best_sample_index = np.argmin(mse_list)
        worst_sample_index = np.argmax(mse_list)
        
    elif mae_list is not None:
        best_sample_index = np.argmin(mae_list)
        worst_sample_index = np.argmax(mae_list)
    else:
        best_sample_index = np.random.randint(0, len(predictions))
        worst_sample_index = np.random.randint(0, len(predictions))
    
    print(f'BEST Sample Index: {best_sample_index}, Worst Sample Index: {worst_sample_index}')
    
    for sample_index in [best_sample_index, worst_sample_index]:  
        plt.figure(figsize=(12, 6))

        input = inputs[sample_index]
        prediction = predictions[sample_index]
        target = targets[sample_index]
    
        # create timesteps
        input_timesteps = np.arange(0, input_seq_len)
        future_timesteps = np.arange(input_seq_len, input_seq_len + output_seq_len)
        
        # plot the data
        plt.plot(input_timesteps, input, label='Historical Data', color='green', alpha=0.7)
        plt.plot(future_timesteps, prediction, label='Predictions', color='red', alpha=0.7)
        plt.plot(future_timesteps, target, label='True Values', color='blue', alpha=0.7)
        
        # add vertical line
        plt.axvline(x=input_seq_len, color='gray', linestyle='--', alpha=0.5)
        
        # set the title and labels
        plt.legend()
        plt.title('Time Series Prediction')
        plt.xlabel('Time Steps (hours)')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        
        # add text
        plt.text(input_seq_len/2, plt.ylim()[1], 'Historical', ha='center')
        plt.text(input_seq_len + output_seq_len/2, plt.ylim()[1], 'Future', ha='center')
        
        # save the plot
        if sample_index == best_sample_index:
            plot_save_path = os.path.join(plot_save_dir, f'best_predictions_{model_suffix}.png')
        else:
            plot_save_path = os.path.join(plot_save_dir, f'worst_predictions_{model_suffix}.png')
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path)
        
        plt.close()

def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plot the training and validation loss.
    
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        save_path (str): Path to save the plot.
    """
    # create a new figure
    plt.figure(figsize=(10, 6))

    # plot the training loss
    plt.plot(train_losses, label="Train Loss", color="blue", alpha=0.7)

    # plot the validation loss
    plt.plot(val_losses, label="Validation Loss", color="red", alpha=0.7)

    # set the title and labels
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Loss plot saved to {save_path}")
        
def save_results(args, mse, mae, result_path='./results/'):
    """
    Save the evaluation results to a CSV file.
    
    Args:
        args (Namespace): Arguments.
        mse (float): Mean Squared Error.
        mae (float): Mean Absolute Error.
        result_path (str): Path to save the results.
    """
    # 准备新数据
    result_path = os.path.join(result_path, f'results.csv')
    
    new_data = {
        'output_seq_len': [args.output_seq_len],
        'seed': [args.seed],
        'mse': [mse],
        'mae': [mae],
        'timestamp': [pd.Timestamp.now()],
        'model_shuffix': [args.model_suffix]
    }
    new_df = pd.DataFrame(new_data)
    
    # 检查文件是否存在
    if os.path.exists(result_path):
        # 追加模式
        existing_df = pd.read_csv(result_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # 创建新文件
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        updated_df = new_df
    
    # 保存结果
    updated_df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")