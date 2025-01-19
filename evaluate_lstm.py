import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.lstm import get_model
# from models.lstm.lstm import Seq2seqLSTM
from utils.data_utils import TimeSeriesDataset
from torch.utils.data import DataLoader
from utils.utils import set_seed, calculate_model_metrics, plot_predictions, save_results
from sklearn.preprocessing import StandardScaler


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
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.csv')
    parser.add_argument('--val_data_path', type=str, default='./data/val_data.csv')
    parser.add_argument('--test_data_path', type=str, default='./data/original/test_data.csv')
    parser.add_argument('--model_path', type=str, default='./save/lstm/')
    parser.add_argument('--plot_path', type=str, default='./results/lstm/')
    parser.add_argument('--result_path', type=str, default='./results/lstm')
    parser.add_argument('--model_suffix', type=str, default='')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()

def main():
    args = arg_parser()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(args.model_path, f'{args.model_suffix}/lstm_model_best_{args.model_suffix}.pth')
    
    # load data
    train_data = pd.read_csv(args.train_data_path)
    train_dataset = TimeSeriesDataset(
        data=train_data,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        is_training=True
    ) 
    test_data = pd.read_csv(args.test_data_path)
    test_dataset = TimeSeriesDataset(
        data=test_data,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        feature_scaler=train_dataset.feature_scaler,
        is_training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # load model
    model = get_model(
        model_type=args.model_type,
        input_feature_len=args.input_feature_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_seq_len
    ).to(device)
    
    # model = Seq2seqLSTM(
    #     input_feature_len=args.input_feature_len,
    #     hidden_size=args.hidden_size,
    #     num_layers=args.num_layers,
    #     output_size=args.output_seq_len
    # ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    
    # evaluate model
    mse, mae, predictions, targets, inputs, mse_list, _ = calculate_model_metrics(model, test_loader, train_dataset.feature_scaler, device)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')
    
    # visualize predictions
    plot_predictions(
        inputs=inputs, 
        predictions=predictions,
        targets=targets,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        mse_list=mse_list,
        save_path=args.plot_path,
        model_suffix=args.model_suffix,
        scaler=train_dataset.feature_scaler
    )
    
    # save results
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    save_results(
        args=args,
        mse=mse,
        mae=mae,
        result_path=args.result_path
    )


if __name__ == '__main__':
    main()