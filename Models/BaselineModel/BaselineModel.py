import pickle
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np
import pandas as pd


class BaselineModel:
    def __init__(self, dataset, scalers_path, max_iters=3000):
        self.dataset = dataset
        # Initialize models for each continuous and categorical target
        self.continuous_models = [LinearRegression() for _ in range(dataset[0][1].shape[0])]
        self.categorical_models = [LogisticRegression(max_iter=max_iters) for _ in range(len(dataset[0][2]))]
        self.continuous_label_names = dataset.continuous_label_names
        self.categorical_label_names = dataset.categorical_label_names

        with open(scalers_path, "rb") as file:
            self.scalers = pickle.load(file)
    
    def train(self, batch_size=32):
        # Prepare data loader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        X_cont = []
        Y_cont = []
        X_cat = []
        Y_cat = [[] for _ in range(len(self.categorical_models))]

        for sequences, cont_target_tensor, cat_target_tensors in dataloader:
            # Use only the last pitch in the sequence as input
            last_pitches = sequences[:, -1, :].numpy()

            X_cont.append(last_pitches)
            Y_cont.append(cont_target_tensor.numpy())

            for i, cat_target_tensor in enumerate(cat_target_tensors):
                Y_cat[i].append(cat_target_tensor.numpy())
        
        X_cont = np.concatenate(X_cont)
        Y_cont = np.concatenate(Y_cont)

        X_cat = X_cont.copy()  # Same input data for categorical targets

        # Train each model on the corresponding target
        for i, model in enumerate(self.continuous_models):
            model.fit(X_cont, Y_cont[:, i])
        
        for i, model in enumerate(self.categorical_models):
            Y_cat_combined = np.concatenate(Y_cat[i])
            model.fit(X_cat, Y_cat_combined.argmax(axis=1))  # Train on the class index, not the one-hot encoding
    
    def predict(self, sequences, scale=False):
        # Extract the last pitch in each sequence
        last_pitches = sequences[:, -1, :].numpy()

        # Predict continuous targets
        cont_preds = [model.predict(last_pitches) for model in self.continuous_models]
        cont_preds = np.stack(cont_preds, axis=-1)

        # Predict categorical targets
        cat_preds = [model.predict_proba(last_pitches) for model in self.categorical_models]
        cat_preds = np.concatenate(cat_preds, axis=-1)

        # Combine continuous and categorical predictions
        all_preds = np.concatenate([cont_preds, cat_preds], axis=-1)

        # Convert predictions to a pandas DataFrame
        flat_cat_names = [name for sublist in self.categorical_label_names for name in sublist]
        col_names = self.continuous_label_names + flat_cat_names
        preds_df = pd.DataFrame(all_preds, columns=col_names)

        if scale:
            # Re-scale continuous predictions
            for i, column in enumerate(self.continuous_label_names):
                if column in self.scalers:
                    scaler = self.scalers[column]
                    preds_df[column] = (preds_df[column] * scaler.scale_) + scaler.mean_

        return preds_df