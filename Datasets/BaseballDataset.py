import torch
from torch.utils.data import Dataset
import json
import pandas
import numpy as np


class BaseballDataset(Dataset):
    def __init__(self, data, config_path, sequence_length,seed=42):
        self.seed = seed
        self.set_seed()
        self.config = self.load_config(config_path)


        self.data = self.add_mask_dimensions(data)
        
        
        self.sequence_length = sequence_length

        self.label_columns = self.get_label_columns()
        self.metadata_columns = self.get_metadata_columns()
        self.categorical_columns = self.get_categorical_columns()

        self.mean_values = self.get_mean_values()

        self.processed_pitches = []
        self.pitch_metadata = []
        self.sequences = []
        self.process_all_pitches()

        self.continuous_label_indices, self.categorical_label_indices, self.continuous_label_names, self.categorical_label_names = self.get_label_indices()
        self.all_label_indices = torch.cat((self.continuous_label_indices, *self.categorical_label_indices))

        self.mask = self.create_mask()
        self.prepare_sequences()
        self.convert_all_pitches_to_tensor()
 
        
    
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
        sample_pitch = self.processed_pitches[0]
        categorical_label_indices = []
        categorical_label_names = []
        continuous_label_indices = []
        continuous_label_names = []

        for key in self.label_columns:
            if key in self.categorical_columns:
                indices = []
                names = []
                for idx, col in enumerate(sample_pitch):
                    if col.startswith(key):
                        indices.append(idx)
                        names.append(col)
                categorical_label_indices.append(torch.LongTensor(indices))
                categorical_label_names.append(names)
            else:
                for idx, col in enumerate(sample_pitch):
                    if col == key:
                        continuous_label_indices.append(idx)
                        continuous_label_names.append(col)
                        
        return torch.LongTensor(continuous_label_indices), categorical_label_indices, continuous_label_names, categorical_label_names
    

    
    def process_all_pitches(self):
        for index, row in self.data.iterrows():
            pitch_data, pitch_metadata = self.process_pitch(row)
            self.processed_pitches.append(pitch_data)
            self.pitch_metadata.append(pitch_metadata)
    
    def convert_all_pitches_to_tensor(self):
        for i in range(len(self.processed_pitches)):
            self.processed_pitches[i] = self.pitch_to_tensor(self.processed_pitches[i])
    
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
    
    
    def pitch_to_tensor(self, pitch):
        # Convert a single pitch dictionary to a tensor
        return torch.tensor(list(pitch.values()), dtype=torch.float)
    

    def create_mask(self):
        mask = torch.zeros(len(self.processed_pitches[0]), dtype=torch.float)
        for idx,name in zip(self.continuous_label_indices,self.continuous_label_names):
            mask[idx] = self.mean_values[name]
        for cat_indices,cat_names in zip(self.categorical_label_indices,self.categorical_label_names):
            for idx,name in zip(cat_indices,cat_names):
                if name.endswith('_mask'):
                    mask[idx] = 1
                else:
                    mask[idx] = 0  
        return mask


    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]


        sequence = []
        for i in sequence_indices:
            sequence.append(self.processed_pitches[i].clone())
  

        #target is unmasked last pitch in sequence
        target = sequence[-1].clone()
        

        # Mask the last pitch in the sequence
        sequence[-1][self.all_label_indices] = self.mask[self.all_label_indices]

        # Convert to tensor
        sequence_tensor = torch.stack(sequence)


        cont_target_tensor = torch.index_select(target,0,self.continuous_label_indices)

        cat_target_tensors = []
        for cat_indices_tensor in self.categorical_label_indices:
            cat_target_tensors.append(torch.index_select(target,0,cat_indices_tensor))
 
        
        return sequence_tensor, cont_target_tensor, cat_target_tensors
    
    
