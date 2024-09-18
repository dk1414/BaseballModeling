import torch
from torch.utils.data import Dataset
import json
import pandas
import numpy as np
import random


class BaseballDataset(Dataset):
    def __init__(self, data, config_path, sequence_length,seed=42):
        self.seed = seed
        self.set_seed()
        self.config = self.load_config(config_path)

        self.data = self.fix_hit_loc(data)
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

        self.pitch_col_names = self.get_pitch_col_names()

        self.continuous_label_indices, self.categorical_label_indices, self.continuous_label_names, self.categorical_label_names, self.mask_indices = self.get_label_indices()
        self.label_and_mask_indices = torch.cat((self.continuous_label_indices, *self.categorical_label_indices, self.mask_indices))

        self.mask = self.create_mask()
        self.prepare_sequences()
        self.processed_pitches = torch.stack(self.processed_pitches)
 
        
    
    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def get_label_columns(self):
        return [column for column, settings in self.config.items() if settings.get('label', False)]
    
    def get_metadata_columns(self):
        return {column for column, settings in self.config.items() if settings.get('metadata', False)}
    
    def get_categorical_columns(self):
        return [column for column, settings in self.config.items() if settings.get('categorical', False)]
    
    def get_mean_values(self):
        continuous_label_columns = list(set(self.label_columns) - set(self.categorical_columns))
        return self.data[continuous_label_columns].mean().to_dict()
    
    def fix_hit_loc(self, data):

        #quick fix for hit_location 2 issue where in the processed data, a stikeout has hit_location_2 true but we expect hit_location_0 to be true

        #this should do it
        data.loc[data['events_strikeout'] == True, 'hit_location_0.0'] = True
        data.loc[data['events_strikeout'] == True, 'hit_location_2.0'] = False

        return data


    
    def add_mask_dimensions(self, data):
        config = self.config
        for column in config:
            if config[column].get('label', False) and config[column].get('categorical', False):
                mask_column = f"{column}_mask"
                data[mask_column] = 0
        return data
    
    def get_label_indices(self):
        col_names = self.pitch_col_names

        categorical_label_indices = []
        categorical_label_names = []

        continuous_label_indices = []
        continuous_label_names = []

        mask_indices = []


        for key in self.label_columns:
            if key in self.categorical_columns:
                indices = []
                names = []
                for idx, col in enumerate(col_names):
                    if col.startswith(key):
                        if col.endswith('_mask'):
                            mask_indices.append(idx)
                        else:
                            indices.append(idx)
                            names.append(col)

                categorical_label_indices.append(torch.LongTensor(indices))
                categorical_label_names.append(names)
            else:
                for idx, col in enumerate(col_names):
                    if col == key:
                        continuous_label_indices.append(idx)
                        continuous_label_names.append(col)
                        
        return torch.LongTensor(continuous_label_indices), categorical_label_indices, continuous_label_names, categorical_label_names, torch.LongTensor(mask_indices)
    

    
    def process_all_pitches(self):
        for idx, row in self.data.iterrows():
            pitch_data, pitch_metadata = self.process_pitch(row)
            self.processed_pitches.append(torch.tensor(pitch_data, dtype=torch.float))
            self.pitch_metadata.append(pitch_metadata)
    
    
    def prepare_sequences(self):
        grouped = self.data.groupby('batter')
        
        for batter, group in grouped:
            group = group.sort_values(by=['game_date', 'at_bat_number', 'pitch_number'])
            indices = group.index.tolist()
            
            for i in range(len(indices) - self.sequence_length):
                sequence_indices = indices[i:i + self.sequence_length]
                self.sequences.append(torch.LongTensor(sequence_indices))


    def get_pitch_col_names(self):
        col_names = []
        for idx, row in self.data.iterrows():
            for key, value in row.items():
                if key in self.metadata_columns:
                    pass
                else:
                    col_names.append(key)
            break
        return col_names
    
    def process_pitch(self, pitch):
        pitch_data = []
        pitch_metadata = []
        
        for key, value in pitch.items():
            if key in self.metadata_columns:
                pitch_metadata.append(value)
            else:
                pitch_data.append(value)
        
        return pitch_data, pitch_metadata
    
    
    def create_mask(self):
        example_pitch = self.pitch_col_names
        mask = torch.zeros(len(example_pitch), dtype=torch.float)

        for idx,col in enumerate(example_pitch):
            if col.endswith('_mask'):
                mask[idx] = 1
            elif col in self.categorical_label_names:
                mask[idx] = 0
            elif col in self.continuous_label_names:
                mask[idx] = self.mean_values[col]
        return mask
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]


        sequence = torch.index_select(self.processed_pitches, 0, sequence_indices).clone()
        

        #target is unmasked last pitch in sequence
        target = sequence[-1].clone()
        

        # Mask the last pitch in the sequence
        sequence[-1][self.label_and_mask_indices] = self.mask[self.label_and_mask_indices]


        cont_target_tensor = torch.index_select(target,0,self.continuous_label_indices)

        cat_target_tensors = []
        for cat_indices_tensor in self.categorical_label_indices:
            cat_target_tensors.append(torch.index_select(target,0,cat_indices_tensor))
 
        
        return sequence, cont_target_tensor, cat_target_tensors
    
    
