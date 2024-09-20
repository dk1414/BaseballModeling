import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append('../')
from DataProc.DataProcessor import DataProcessor



def main():

    # data = pd.read_csv("statcast_2015-2024.csv")
    # data_proc = DataProcessor(data,"configv3.json")
    # d = data_proc.get_processed_data()
    # d.to_csv('full_cleaned_94.csv')
    # data_proc.save_scalers('full_scalers_94.pkl')


    raw_data = pd.read_csv("full_cleaned_94.csv")
    start_date = pd.to_datetime("2017-01-01")
    end_date = pd.to_datetime("2018-01-01")
    raw_data['game_date'] = pd.to_datetime(raw_data['game_date'])
    total_days = (end_date - start_date).days
    split_date = pd.to_datetime(start_date) + timedelta(days=int(total_days * 0.6))

    # Split data into train and validation sets
    train = raw_data[(raw_data['game_date'] > start_date) & (raw_data['game_date'] < split_date)]
    valid = raw_data[(raw_data['game_date'] > split_date) & (raw_data['game_date'] < end_date)]
    train.to_csv(f"med_train_94.csv", index=False)
    valid.to_csv(f"med_test_94.csv", index=False)


if __name__ == "__main__":
    #main()

    print(pd.read_csv('med_train.csv').head(10))


    

