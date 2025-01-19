import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.lstm import get_model
# from models.lstm.lstm import Seq2seqLSTM
# from models.lstm.lstm_improve import Seq2seqLSTMWithCNN, Seq2seqLSTMWithAttention, Seq2seqLSTMWithAttentionCNN
from utils.data_utils import TimeSeriesDataset
from torch.utils.data import DataLoader
from utils.utils import set_seed, warmup_cosine, plot_loss

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate(model, data_loader, device='cuda'):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target[:, :, -1])
            total_loss += loss.item()
    return total_loss / len(data_loader)
    

def train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, save_path='./save/lstm/best_model.pth', device='cuda'):
    """
    Train the LSTM model.
    
    Args:
        model (nn.Module): LSTM model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4)
    
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    cnt = 0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        # warmup_cosine(
        #     optimizer=optimizer , 
        #     current_epoch=epoch , 
        #     max_epoch=num_epochs ,
        #     lr_min=0.01 * learning_rate , 
        #     lr_max=learning_rate ,
        #     warmup_epoch=0.1*num_epochs
        # )
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            # Sechdule sampling
            output = model(data, target)
            loss = criterion(output, target[:, :, -1])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        # Evaluation
        val_loss = evaluate(model, val_loader, device)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Save model to {save_path}')
            cnt = 0
        else:
            cnt += 1
            if cnt == 12:
                break
        # Update learning rate
        scheduler.step(val_loss)
    
    return train_loss_list, val_loss_list


def arg_parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="lstm", 
                        choices=["lstm", "lstm_attention", "lstm_cnn", "lstm_attention_cnn"], 
                        help="LSTM model type.")
    parser.add_argument('--input_seq_len', type=int, default=96)
    parser.add_argument('--output_seq_len', type=int, default=96)
    parser.add_argument('--input_feature_len', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.csv')
    parser.add_argument('--val_data_path', type=str, default='./data/val_data.csv')
    parser.add_argument('--save_path', type=str, default='./save/lstm/')
    parser.add_argument('--model_suffix', type=str, default='')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


def main():
    args = arg_parser()
    
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_path = os.path.join(args.save_path, args.model_suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model_save_path = os.path.join(save_path, f'lstm_model_best_{args.model_suffix}.pth')
    loss_plot_save_path = os.path.join(save_path, f'loss_{args.model_suffix}.png')
    loss_csv_save_path = os.path.join(save_path, f'loss_{args.model_suffix}.csv')
    
    train_data = pd.read_csv(args.train_data_path)
    val_data = pd.read_csv(args.val_data_path)
    
    train_dataset = TimeSeriesDataset(
        data=train_data, 
        input_seq_len=args.input_seq_len, 
        output_seq_len=args.output_seq_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TimeSeriesDataset(
        data=val_data, 
        input_seq_len=args.input_seq_len, 
        output_seq_len=args.output_seq_len,
        feature_scaler=train_dataset.feature_scaler,
        is_training=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = get_model(
        model_type=args.model_type,
        input_feature_len=args.input_feature_len,
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        output_size=args.output_seq_len
    )
    
    # model = Seq2seqLSTM(
    #     input_feature_len=args.input_feature_len,
    #     hidden_size=args.hidden_size, 
    #     num_layers=args.num_layers,
    #     output_size=args.output_seq_len
    # )
    
    train_losses, val_losses = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate,
        save_path=model_save_path
    )
    
    plot_loss(train_losses=train_losses , val_losses= val_losses , save_path=loss_plot_save_path)
    loss_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv(loss_csv_save_path, index=False)

if __name__ == '__main__':
    main()