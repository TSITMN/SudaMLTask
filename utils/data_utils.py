import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_seq_len, output_seq_len, feature_scaler=None, is_training=True):
        """
        Dataset for time series data with preprocessing embedded.

        Args:
            data (pd.DataFrame): Raw data.
            input_seq_len (int): Length of input sequences.
            output_seq_len (int): Length of output sequences.
            scaler (StandardScaler or None): Scaler object for standardization. If None and is_training=True, a new scaler is created.
            is_training (bool): Whether the dataset is for training. If False, uses the provided scaler.
        """
        target_col = 'cnt'
        drop_columns = ['instant', 'dteday', 'yr', 'casual', 'registered']
        feature = data.drop(columns=drop_columns)
        target = data[target_col]
        
        feature = feature.astype('float32')
        target = target.astype('float32')

        if is_training:
            self.feature_scaler = StandardScaler()
            feature_scaled = self.feature_scaler.fit_transform(feature)       
        else:
            self.feature_scaler = feature_scaler
            feature_scaled = self.feature_scaler.transform(feature)
        # print("Feature scaled:", feature_scaled)
        # print("Feature scaled's shape:", feature_scaled.shape)
        # print("Feature scaler:", self.feature_scaler.mean_, self.feature_scaler.var_)

        self.x, self.y = self._create_sequences(feature_scaled, input_seq_len, output_seq_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _create_sequences(self, feature, input_seq_len, output_seq_len):
        """
        Create sliding window sequences for input and output.

        Args:
            feature(numpy array): Scaled feature data.
            target(numpy array): Scaled target data.
            input_seq_len (int): Length of input sequences.
            output_seq_len (int): Length of output sequences.

        Returns:
            tuple: Input sequences (x) and output sequences (y).
        """
        x, y = [], []
        for i in range(len(feature) - input_seq_len - output_seq_len + 1):
            input_seq = feature[i:i + input_seq_len]
            # output_seq = target[i + input_seq_len:i + input_seq_len + output_seq_len]
            output_seq = feature[i + input_seq_len:i + input_seq_len + output_seq_len]
            x.append(input_seq)
            y.append(output_seq)
        return np.array(x), np.array(y)


def split_train_val(data_path, new_train_path, new_val_path, val_ratio=0.2):
    """
    Split the original training data into training and validation sets.
    
    Args:
        data_path (str): Path to the original training data.
        new_train_path (str): Path to save the new training data.
        new_val_path (str): Path to save the new validation data.
        val_ratio (float): Ratio of validation data.
    """
    data = pd.read_csv(data_path)
    val_size = int(len(data) * val_ratio)
    val_data = data[-val_size:]
    train_data = data[:-val_size]
    val_data.to_csv(new_val_path, index=False)
    train_data.to_csv(new_train_path, index=False)
    
    print(f"Training data saved to {new_train_path}")
    print(f"Validation data saved to {new_val_path}")
    
    

def test():
    file_path = '../data/original/train_data.csv'
    new_train_path = '../data/train_data.csv'
    new_val_path = '../data/val_data.csv'
    
    data = pd.read_csv(file_path)
    split_train_val(file_path, new_train_path, new_val_path, val_ratio=0.1)


if __name__ == "__main__":

    test()
