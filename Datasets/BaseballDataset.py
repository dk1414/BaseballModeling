import torch
from torch.utils.data import Dataset
from torch.masked import masked_tensor
import json
import pandas
import numpy as np


class BaseballDataset(Dataset):
    def __init__(self, data, config_path, sequence_length, encode_pos=False, masked_tensor=False, seed=42):
        self.seed = seed
        self.set_seed()
        self.encode_pos = encode_pos
        self.masked_tensor = masked_tensor
        self.config = self.load_config(config_path)

        self.data = data
        if not self.masked_tensor:
            self.data = self.add_mask_dimensions(data)
        
        
        self.sequence_length = sequence_length
        self.label_columns = self.get_label_columns()
        self.metadata_columns = self.get_metadata_columns()
        self.categorical_columns = self.get_categorical_columns()
        self.mean_values = self.get_mean_values()
        self.processed_pitches = []
        self.sequences = []
        self.process_all_pitches()
        self.continuous_label_indices, self.categorical_label_indices = self.get_label_indices()

        if self.masked_tensor:
            self.mask = self.create_mask()

        self.prepare_sequences()
    
    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def get_label_columns(self):
        return {column for column, settings in self.config.items() if settings.get('label', False)}
    
    def get_metadata_columns(self):
        return {column for column, settings in self.config.items() if settings.get('metadata', False)}
    
    def get_categorical_columns(self):
        return {column for column, settings in self.config.items() if settings.get('categorical', False)}
    
    def get_mean_values(self):
        continuous_label_columns = list(self.label_columns - self.categorical_columns)
        return self.data[continuous_label_columns].mean().to_dict()
    
    def add_mask_dimensions(self, data):
        config = self.config
        for column in config:
            if config[column].get('label', False) and config[column].get('categorical', False):
                mask_column = f"{column}_mask"
                data[mask_column] = 0
        return data
    
    def get_label_indices(self):
        sample_pitch = self.processed_pitches[0][0]
        categorical_label_indices = []
        continuous_label_indices = []

        for key in self.label_columns:
            if key in self.categorical_columns:
                for idx, col in enumerate(sample_pitch):
                    if col.startswith(key):
                        categorical_label_indices.append(idx)
            else:
                for idx, col in enumerate(sample_pitch):
                    if col == key:
                        continuous_label_indices.append(idx)
                        
        return continuous_label_indices, categorical_label_indices
    
    def create_mask(self):

        pitch_dim = len(self.processed_pitches[0][0])
        sequence_mask = torch.ones((self.sequence_length,pitch_dim))

        for i in range(pitch_dim):
            if i in self.continuous_label_indices or i in self.categorical_label_indices:
                sequence_mask[-1][i] = 0
        
        return sequence_mask == 1
            
    
    def process_all_pitches(self):
        for index, row in self.data.iterrows():
            pitch_data, pitch_metadata = self.process_pitch(row)
            self.processed_pitches.append((pitch_data, pitch_metadata))
    
    def prepare_sequences(self):
        grouped = self.data.groupby('batter')
        
        for batter, group in grouped:
            group = group.sort_values(by=['game_date', 'at_bat_number'])
            indices = group.index.tolist()
            
            for i in range(len(indices) - self.sequence_length):
                sequence_indices = indices[i:i + self.sequence_length]
                self.sequences.append(sequence_indices)

 
    
    def process_pitch(self, pitch):
        pitch_data = {}
        pitch_metadata = {}
        
        for key, value in pitch.items():
            if key in self.metadata_columns:
                pitch_metadata[key] = value
            else:
                pitch_data[key] = value
        
        return pitch_data, pitch_metadata
    
    def mask_values(self, pitch):

        if not self.masked_tensor: #fill w mean/mode

            for key in self.label_columns:
                if key in self.categorical_columns:
                    for col in pitch:
                        if col.startswith(key):
                            pitch[col] = 0
                    pitch[f'{key}_mask'] = 1
                elif key not in self.metadata_columns:
                    pitch[key] = self.mean_values[key]
            return pitch



    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]


        sequence = []
        metadata = []
        for i in sequence_indices:
            sequence.append(self.processed_pitches[i][0].copy())
            metadata.append(self.processed_pitches[i][1])

        if self.encode_pos:
            # Add positional encoding
            for i, pitch in enumerate(sequence):
                pitch['pos'] = i / self.sequence_length

        #target is unmasked last pitch in sequence
        target = sequence[-1].copy()
        
        if not self.masked_tensor:
            # Mask the last pitch in the sequence
            sequence[-1] = self.mask_values(sequence[-1])

        # Convert to tensor
        sequence_tensor = self.sequence_to_tensor(sequence)

        if self.masked_tensor:
            sequence_tensor = masked_tensor(sequence_tensor,self.mask)

        target = self.pitch_to_tensor(target)
        cont_target_tensor = torch.index_select(target,0,torch.LongTensor(self.continuous_label_indices))
        cat_target_tensor = torch.index_select(target,0,torch.LongTensor(self.categorical_label_indices))
        
        return sequence_tensor, cont_target_tensor, cat_target_tensor
    
    def sequence_to_tensor(self, sequence):
        # Convert the list of pitch dictionaries to a tensor
        sequence_tensor = torch.stack([self.pitch_to_tensor(pitch) for pitch in sequence])
        return sequence_tensor
    
    def pitch_to_tensor(self, pitch):
        # Convert a single pitch dictionary to a tensor, excluding metadata columns
        return torch.tensor(list(pitch.values()), dtype=torch.float)