import sys
sys.path.append('../')

from Datasets.BaseballDataset import BaseballDataset

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import pandas as pd
import os
import matplotlib.pyplot as plt



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, hidden_dim, output_dim, sequence_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use the output of the last pitch in the sequence
        x = self.fc_layers(x)
        return x

class CustomLoss(nn.Module):
    def __init__(self, weight_param):
        super(CustomLoss, self).__init__()
        self.weight_param = weight_param

    def forward(self, output, target_continuous, target_categorical):
        # Continuous target loss (MSE)
        mse_loss = F.mse_loss(output[:, :target_continuous.size(1)], target_continuous)
        
        # Categorical target loss (Cross-Entropy)
        cross_entropy_loss = F.cross_entropy(output[:, target_continuous.size(1):], target_categorical.type(torch.float))
        
        # Weighted sum of the losses
        loss = (self.weight_param * mse_loss) + ((1 - self.weight_param) * cross_entropy_loss)
        return loss




# Training and Evaluation functions
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequence_tensor, cont_target_tensor, cat_target_tensor in train_loader:
        sequence_tensor, cont_target_tensor, cat_target_tensor = sequence_tensor.to(device), cont_target_tensor.to(device), cat_target_tensor.to(device)

        optimizer.zero_grad()
        output = model(sequence_tensor)
        loss = criterion(output, cont_target_tensor, cat_target_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequence_tensor, cont_target_tensor, cat_target_tensor in val_loader:
            sequence_tensor, cont_target_tensor, cat_target_tensor = sequence_tensor.to(device), cont_target_tensor.to(device), cat_target_tensor.to(device)
            output = model(sequence_tensor)
            loss = criterion(output, cont_target_tensor, cat_target_tensor)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def plot_loss(train_losses,val_losses, loss_plot_path):

    # Plot and save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(loss_plot_path)

    return



def main(train_config_path):

    # Load configuration from JSON file
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)

    # Extract parameters from the config
    num_heads = train_config["num_heads"]
    num_encoder_layers = train_config["num_encoder_layers"]
    hidden_dim = train_config["hidden_dim"]
    sequence_length = train_config["sequence_length"]
    dropout = train_config["dropout"]
    batch_size = train_config["batch_size"]
    loss_weight_param = train_config["weight_param"]
    num_epochs = train_config["num_epochs"]
    model_save_dir = train_config["model_save_dir"]
    model_save_path = os.path.join(model_save_dir, 'transformer_model.pth')
    loss_plot_path = os.path.join(model_save_dir, 'loss_plot.png')


    # File paths
    config_path = train_config["config_path"]
    train_path = train_config["train_data_path"]
    valid_path = train_config["valid_data_path"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Data")

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)

    print(f"Train Shape: {train_data.shape}")
    print(f"Valid Shape: {valid_data.shape}")

    print("Creating Datasets")

    #Create datasets
    train_dataset = BaseballDataset(train_data,config_path,sequence_length)
    valid_dataset = BaseballDataset(valid_data,config_path,sequence_length)
    
    print("Creating Dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6)


    # Hyperparameters
    input_dim = train_dataset[0][0].shape[1]  # Number of features in a single pitch
    output_dim = train_dataset[0][1].shape[0] + train_dataset[0][2].shape[0] 
    

    # Initialize the model, loss function, and optimizer
    model = TransformerModel(input_dim, num_heads, num_encoder_layers, hidden_dim, output_dim, sequence_length, dropout)
    criterion = CustomLoss(loss_weight_param)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Starting epoch: {epoch}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    

    # Create directory to save model and plot
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_loss(train_losses,val_losses,loss_plot_path)
    print(f"Loss Plot saved to {loss_plot_path}")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
