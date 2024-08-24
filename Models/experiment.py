import sys
sys.path.append('../')

from Datasets.BaseballDataset import BaseballDataset

import copy
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
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Linear layer for the residual connection
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

        self._init_weights()

    def forward(self, x):

        x_emb = self.embedding(x)
        x_emb = self.positional_encoding(x_emb)
        x_transformed = self.transformer_encoder(x_emb)
        
        # Use the output of the last pitch in the sequence
        x_last = x_transformed[:, -1, :]
        
        # Extract the last element of the original input
        x_last_input = x[:, -1, :]
        
        # Pass the last element through the residual fc layer
        x_last_input_fc = self.residual_fc(x_last_input)
        
        # Concatenate the features
        x_combined = torch.cat((x_last, x_last_input_fc), dim=-1)
        
        # Pass combined output through fc_layers
        x_out = self.fc_layers(x_combined)
        
        return x_out
    

    def _init_weights(self):
        # Apply Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                nn.init.xavier_uniform_(m.linear1.weight)
                nn.init.xavier_uniform_(m.linear2.weight)
                nn.init.constant_(m.linear1.bias, 0)
                nn.init.constant_(m.linear2.bias, 0)

class CustomLoss(nn.Module):
    def __init__(self, weight_param):
        super(CustomLoss, self).__init__()
        self.weight_param = weight_param

    def forward(self, output, target_continuous, target_categorical):
        # Continuous target loss (MSE)
        mse_loss = F.mse_loss(output[:, :target_continuous.size(1)], target_continuous)
        
        # Categorical target loss (Cross-Entropy) for each categorical feature
        cross_entropy_loss = 0
        start_idx = target_continuous.size(1)
        for cat_target in target_categorical:
            end_idx = start_idx + cat_target.size(1)
            cross_entropy_loss += F.cross_entropy(output[:, start_idx:end_idx], cat_target)
            start_idx = end_idx
        

        # Weighted sum of the losses
        loss = (self.weight_param * mse_loss) + ((1 - self.weight_param) * cross_entropy_loss)
        return loss




# Training and Evaluation functions
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequence_tensor, cont_target_tensor, cat_target_tensor in train_loader:
        sequence_tensor, cont_target_tensor = sequence_tensor.to(device), cont_target_tensor.to(device)
        cat_targets = [t.to(device) for t in cat_target_tensor]

        optimizer.zero_grad()
        output = model(sequence_tensor)
        loss = criterion(output, cont_target_tensor, cat_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequence_tensor, cont_target_tensor, cat_target_tensor in val_loader:
            sequence_tensor, cont_target_tensor = sequence_tensor.to(device), cont_target_tensor.to(device)
            cat_targets = [t.to(device) for t in cat_target_tensor]
            output = model(sequence_tensor)
            loss = criterion(output, cont_target_tensor, cat_targets)
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
    learning_rate = train_config["learning_rate"]
    num_epochs = train_config["num_epochs"]
    num_workers = train_config["num_workers"]
    model_save_dir = train_config["model_save_dir"]


    # File paths
    config_path = train_config["config_path"]
    train_path = train_config["train_data_path"]
    valid_path = train_config["valid_data_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)
    

    print("Loading Data", flush=True)

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)

    print(f"Train Shape: {train_data.shape}", flush=True)
    print(f"Valid Shape: {valid_data.shape}", flush=True)

    print("Creating Datasets", flush=True)

    #Create datasets
    train_dataset = BaseballDataset(train_data,config_path,sequence_length)
    valid_dataset = BaseballDataset(valid_data,config_path,sequence_length)

    print(f"Train Label Cols: {train_dataset.continuous_label_indices}, {train_dataset.categorical_label_indices}")
    print(f"Valid Label Cols: {valid_dataset.continuous_label_indices}, {valid_dataset.categorical_label_indices}")
    
    print("Creating Dataloaders", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Hyperparameters
    train_config["input_dim"] = train_dataset[0][0].shape[1]  # Number of features in a single pitch
    input_dim = train_config["input_dim"]
    train_config["output_dim"] = train_dataset[0][1].shape[0] + sum([t.shape[0] for t in train_dataset[0][2]]) 
    output_dim = train_config["output_dim"]



    #grid search (a lot of these will probably only be 1 or 2 values)

    for num_head in num_heads:
        for num_encoder_layer in num_encoder_layers:
            for h_dim in hidden_dim:
                for drop in dropout:
                    for loss_param in loss_weight_param:
                        for lr in learning_rate:
                            for epochs in num_epochs:

                                print(f"Starting Experiment: nheads-{num_head}, nencoder-{num_encoder_layer}, hdim-{h_dim}, dropout-{drop}, loss weight-{loss_param}, lr-{lr}, epochs-{epochs}", flush=True)

                                # Initialize the model, loss function, and optimizer
                                model = nn.DataParallel(TransformerModel(input_dim, num_head, num_encoder_layer, h_dim, output_dim, sequence_length, drop))


                                print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
                                
                                criterion = CustomLoss(loss_param)
                                optimizer = optim.Adam(model.parameters(), lr=lr)

                                model.to(device)

                                train_losses = []
                                val_losses = []

                                for epoch in range(epochs):
                                    print(f"Starting epoch: {epoch}", flush=True)
                                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                                    val_loss = evaluate(model, val_loader, criterion, device)
                                    train_losses.append(train_loss)
                                    val_losses.append(val_loss)
                                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}", flush=True)
                                
                                curr_dir = model_save_dir + f"/h{num_head}_e{num_encoder_layer}_h{h_dim}_d{drop}_lp{loss_param}_lr{lr}_ep{epochs}"
                                # Create directory to save model and plot
                                os.makedirs(curr_dir, exist_ok=True)

                                model_save_path = os.path.join(curr_dir, 'transformer_model.pth')
                                loss_plot_path = os.path.join(curr_dir, 'loss_plot.png')
                                
                                # Save the model
                                torch.save(model.module.state_dict(), model_save_path)
                                print(f"Model saved to {model_save_path}", flush=True)

                                plot_loss(train_losses,val_losses,loss_plot_path)
                                print(f"Loss Plot saved to {loss_plot_path}", flush=True)

                                # Create a copy of train_config and update it with the specific hyperparameters for this run
                                specific_train_config = copy.deepcopy(train_config)
                                specific_train_config["num_heads"] = num_head
                                specific_train_config["num_encoder_layers"] = num_encoder_layer
                                specific_train_config["hidden_dim"] = h_dim
                                specific_train_config["dropout"] = drop
                                specific_train_config["weight_param"] = loss_param
                                specific_train_config["learning_rate"] = lr
                                specific_train_config["num_epochs"] = epochs

                                # Save the specific config for this run
                                config_save_path = os.path.join(curr_dir, 'model_config.json')
                                with open(config_save_path, 'w') as f:
                                    json.dump(specific_train_config, f, indent=4)
                                print(f"Config saved to {config_save_path}", flush=True)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python experiment.py <path_to_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
