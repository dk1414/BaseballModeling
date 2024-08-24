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
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



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

class TransformerHelper:
    def __init__(self,model_path, config_path):

        self.model = self.load_model(model_path,config_path)


    def load_model(self, model_path, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)

        model = TransformerModel(
            input_dim=config['input_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            sequence_length=config['sequence_length'],
            dropout=config.get('dropout', 0.1)  # Optional: provide a default value for dropout if not in config
        )

        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model



    def make_preds(self, dataset, scaler_path, device, batch_size, scale=False):
        model = self.model
        model.to(device)
        #get column names in correct order
        flat_cat_names = []
        for names in dataset.categorical_label_names:
            flat_cat_names = flat_cat_names + names 
        col_names = dataset.continuous_label_names + flat_cat_names

        #create dataloader for dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()

        preds_array = [] #keep trask of preds for each batch
        true_array = [] #keep track of true values
        with torch.no_grad():
            idx = 0
            for sequence_tensor, cont_target_tensor, cat_target_tensor in loader:
                idx += 1
                if idx % 10 == 0:
                    print(f"Starting Batch: {idx}")
                    
                sequence_tensor, cont_target_tensor = sequence_tensor.to(device), cont_target_tensor.to(device)
                cat_targets = [t.to(device) for t in cat_target_tensor]
                output = model(sequence_tensor)

                #first k logits correspond to continuous outputs, k = cont_target.size(1)
                cont_output = output[:, :cont_target_tensor.size(1)].cpu().squeeze(0).detach().numpy()
                cont_targets = cont_target_tensor.cpu().squeeze(0).detach().numpy()

                #can have multiple kinds of categorical outputs. If cat_targets is (batch_size, 2, 10), there are 2 kinds of cateogorical outputs, each with 10 values.
                #The first 10 logits after the continuous logits will correspond to first categorical output, second 10 to the second, so this requires multiple softmaxes
                cat_probs = []
                cat_target_probs = []
                start_idx = cont_target_tensor.size(1)
                for cat_target in cat_targets:
                    end_idx = start_idx + cat_target.size(1)
                    cat_probs.append(nn.functional.softmax(output[:, start_idx:end_idx],dim=1).cpu().squeeze(0).detach().numpy())
                    cat_target_probs.append(cat_target.cpu().squeeze(0).detach().numpy())
                    start_idx = end_idx
        
                #cat continuous and categorical outputs together
                preds = cont_output
                for probs in cat_probs:
                    preds = np.concatenate((preds, probs),axis=1)
                
                preds_array.append(preds)

                true = cont_targets
                for probs in cat_target_probs:
                    true = np.concatenate((true, probs),axis=1)
                
                true_array.append(true)

        #make single preds pd     
        preds_array = np.vstack(preds_array)
        preds_pd = pd.DataFrame(preds_array, columns=col_names)

        true_array = np.vstack(true_array)
        true_pd = pd.DataFrame(true_array, columns=col_names)

        if scale:
            #scale continuous outputs back to real values
            with open(scaler_path, "rb") as file:
                scalers = pickle.load(file)

            for column, scaler in scalers.items():
                if column in preds_pd:
                    preds_pd[column] = (preds_pd[column] * scaler.scale_) + scaler.mean_
                    true_pd[column] = (true_pd[column] * scaler.scale_) + scaler.mean_
        

        return preds_pd, true_pd