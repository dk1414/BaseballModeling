import pickle
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import mode
import numpy as np
import pandas as pd

class BaselineModel:
    def __init__(self, dataset, scalers_path, max_iters=3000, pred_mode=False, pred_mean=False):
        self.dataset = dataset
        self.pred_mode = pred_mode
        self.pred_mean = pred_mean
        self.continuous_label_names = dataset.continuous_label_names
        self.categorical_label_names = dataset.categorical_label_names

        # Initialize models for each continuous and categorical target
        if not pred_mean:
            self.continuous_models = [LinearRegression() for _ in range(dataset[0][1].shape[0])]
        if not pred_mode:
            self.categorical_models = [LogisticRegression(max_iter=max_iters) for _ in range(len(dataset[0][2]))]

        with open(scalers_path, "rb") as file:
            self.scalers = pickle.load(file)

        # Placeholders for storing means and modes
        self.cont_means = None
        self.cat_modes = None
    

    #this code sucks and is way more complicated that necessary so that we can use this on large datasets and reuse the transformer datasets
    def train(self, batch_size=1000):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        # Initialize lists to hold batch data
        X_batches = []
        Y_cont_batches = []
        Y_cat_batches = [[] for _ in range(len(self.dataset.categorical_label_names))]

        # Initialize final data arrays
        X_final = []
        Y_cont_final = []
        Y_cat_final = [[] for _ in range(len(self.dataset.categorical_label_names))]

        print("Creating matrices")

        batch_counter = 0
        for sequences, cont_target_tensor, cat_target_tensors in dataloader:
            if batch_counter % 10 == 0:
                print(f"Processing batch {batch_counter}")

            # Use only the last pitch in the sequence as input
            last_pitches = sequences[:, -1, :].numpy()
            cont_targets = cont_target_tensor.numpy()
            cat_targets = [cat_tensor.numpy() for cat_tensor in cat_target_tensors]

            # Accumulate batch data
            X_batches.append(last_pitches)
            Y_cont_batches.append(cont_targets)
            for i, cat_target in enumerate(cat_targets):
                Y_cat_batches[i].append(cat_target)

            # Concatenate and append data every 10 batches
            if (batch_counter + 1) % 10 == 0:
                print("Concatenating and appending data")

                # Concatenate current batch data
                X_combined = np.concatenate(X_batches, axis=0)
                Y_cont_combined = np.concatenate(Y_cont_batches, axis=0)
                Y_cat_combined = [np.concatenate(cat_list, axis=0) for cat_list in Y_cat_batches]

                # Append to final data arrays
                X_final.append(X_combined)
                Y_cont_final.append(Y_cont_combined)
                for i in range(len(Y_cat_combined)):
                    Y_cat_final[i].append(Y_cat_combined[i])

                # Clear batch lists to free up memory
                X_batches = []
                Y_cont_batches = []
                Y_cat_batches = [[] for _ in range(len(self.dataset.categorical_label_names))]

            batch_counter += 1

        # Final concatenation if the number of batches is not a multiple of 10
        if len(X_batches) > 0:
            print("Final concatenation")
            X_combined = np.concatenate(X_batches, axis=0)
            Y_cont_combined = np.concatenate(Y_cont_batches, axis=0)
            Y_cat_combined = [np.concatenate(cat_list, axis=0) for cat_list in Y_cat_batches]

            X_final.append(X_combined)
            Y_cont_final.append(Y_cont_combined)
            for i in range(len(Y_cat_combined)):
                Y_cat_final[i].append(Y_cat_combined[i])

        # Convert final lists to numpy arrays
        X = np.concatenate(X_final, axis=0)
        Y_cont = np.concatenate(Y_cont_final, axis=0)
        Y_cat = [np.concatenate(cat_list, axis=0) for cat_list in Y_cat_final]

        print("Finished processing all batches")
        

        print("Starting cont train")
        if self.pred_mean:
            # Calculate and store the means of continuous targets
            self.cont_means = np.mean(Y_cont, axis=0)
        else:
            # Train each model on the corresponding continuous target
            for i, model in enumerate(self.continuous_models):
                model.fit(X, Y_cont[:, i])
        print("Starting cat train")
        if self.pred_mode:
            # Calculate and store the modes of categorical targets
            self.cat_modes = []
            for i in range(len(self.categorical_label_names)):
                mode_value = mode(Y_cat[i].argmax(axis=1)).mode
                self.cat_modes.append(np.eye(Y_cat[i].shape[1])[mode_value])
        else:
            # Train each model on the corresponding categorical target
            for i, model in enumerate(self.categorical_models):
                model.fit(X, Y_cat[i].argmax(axis=1))  # Train on the class index, not the one-hot encoding
    
    def predict(self, sequences, scale=False):
        # Extract the last pitch in each sequence
        last_pitches = sequences[:, -1, :].numpy()

        # Predict continuous targets
        if self.pred_mean:
            # Use precomputed means
            cont_preds = np.tile(self.cont_means, (last_pitches.shape[0], 1))
        else:
            cont_preds = [model.predict(last_pitches) for model in self.continuous_models]
            cont_preds = np.stack(cont_preds, axis=-1)

        # Predict categorical targets
        if self.pred_mode:
            # Use precomputed modes
            cat_preds = np.tile(np.concatenate(self.cat_modes, axis=-1), (last_pitches.shape[0], 1))
        else:
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