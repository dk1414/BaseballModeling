import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

class DataProcessor:
    def __init__(self, raw_data, config_path, one_hot = True, seed=42):
        np.random.seed(seed)

        self.raw_data = raw_data
        self.config = self.load_config(config_path)
        self.selected_columns = list(self.config.keys())
        self.processed_data = self.raw_data[self.selected_columns].copy()
        self.one_hot = one_hot

        self.scalers = {}  # Dictionary to store scalers for each column
        
        print(f"Selected Columns: {self.selected_columns}")
        print(f'Raw Data Shape: {self.raw_data.shape}')


    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def handle_missing_values(self):
        for column, settings in self.config.items():
            if column == 'events':  # Handle 'events' column separately
                events_to_map = set(['caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home', 'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 'passed_ball'])
                self.processed_data[column] = self.raw_data.apply(
                    lambda row: row['type'] if pd.isnull(row[column]) or row[column] in events_to_map else row[column], axis=1)

            initial_missing_count = self.processed_data[column].isnull().sum()
            strategy = settings.get('missing_value_strategy', 'drop')
            if strategy == 'drop':
                self.processed_data.dropna(subset=[column], inplace=True)
            elif strategy in ['mean', 'median', 'mode']:
                if strategy == 'mean':
                    fill_value = self.processed_data[column].mean()
                elif strategy == 'median':
                    fill_value = self.processed_data[column].median()
                elif strategy == 'mode':
                    fill_value = self.processed_data[column].mode().iloc[0]
                self.processed_data[column].fillna(fill_value, inplace=True)
            elif isinstance(strategy, (int, float, str)):
                self.processed_data[column].fillna(strategy, inplace=True)

            final_missing_count = self.processed_data[column].isnull().sum()
            handled_count = initial_missing_count - final_missing_count
            print(f"Handled {handled_count} missing values in column '{column}'")
            print(f"Missing values after handling in column '{column}': {final_missing_count}")

    def convert_data_types(self):
        for column, settings in self.config.items():
            if settings.get('categorical', False):
                initial_row_count = self.processed_data.shape[0]
                if 'value_map' in settings:
                    value_map = settings['value_map']
                    self.processed_data[column] = self.processed_data[column].replace(value_map)
                if 'drop_values' in settings:
                    drop_values = settings['drop_values']
                    self.processed_data = self.processed_data[~self.processed_data[column].isin(drop_values)]
                    final_row_count = self.processed_data.shape[0]
                    dropped_rows = initial_row_count - final_row_count
                    print(f"Dropped {dropped_rows} rows for column '{column}' due to drop values")
                self.processed_data[column] = self.processed_data[column].astype('category')
            elif 'datetime_format' in settings:
                self.processed_data[column] = pd.to_datetime(self.processed_data[column], format=settings['datetime_format'])
        
        print(f"Processed Data Shape after convert_data_types: {self.processed_data.shape}")

    def standardize_or_normalize(self):
        scaler = StandardScaler()
        for column, settings in self.config.items():
            if settings.get('standardize', False):
                self.processed_data[[column]] = scaler.fit_transform(self.processed_data[[column]])
                self.scalers[column] = scaler

        print(f"Processed Data Shape after standardize_or_normalize: {self.processed_data.shape}")
    
    def reverse_standard_scaling(self, data):

        reversed_data = data.copy()
        for column, scaler in self.scalers.items():
            reversed_data[column] = (data[column] * scaler.scale_) + scaler.mean_
        return reversed_data

    def one_hot_encode(self):
        categorical_columns = [col for col, settings in self.config.items() if settings.get('categorical', False) and not settings.get('metadata', False)]

        if categorical_columns:
            # Check for NaN values before encoding
            for col in categorical_columns:
                missing_values_count = self.processed_data[col].isnull().sum()
                if missing_values_count > 0:
                    print(f"Column '{col}' has {missing_values_count} missing values before encoding.")
            
            # Perform one-hot encoding using pd.get_dummies
            self.processed_data = pd.get_dummies(data=self.processed_data, columns=categorical_columns, prefix=categorical_columns, drop_first=True)
            

        print(f"Processed Data Shape after one_hot_encode: {self.processed_data.shape}")


    def get_processed_data(self):
        self.handle_missing_values()
        self.convert_data_types()
        self.standardize_or_normalize()
        if self.one_hot:
            self.one_hot_encode()
        print(f'New Data Shape: {self.processed_data.shape}')
        return self.processed_data

