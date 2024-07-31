import pybaseball
import pandas as pd
import sys
sys.path.append('../')
from DataProc.DataProcessor import DataProcessor


def pull_data():

    pybaseball.cache.enable()

    #pull all pitch level data from 2015-2024
    full_data = pybaseball.statcast(start_dt="2015-01-01", end_dt="2024-06-30", verbose=True)

    data_path = 'statcast_2015-2024.csv'
    #saving the data to a CSV file
    full_data.to_csv('statcast_2015-2024.csv', index=False)

    return data_path


def process_data(data_path,config_path):
    raw_data = pd.read_csv(data_path)


    #save 2023-2024 for validation
    train = raw_data[raw_data['game_date'] < "2023-04-01"]
    valid = raw_data[raw_data['game_date'] >= "2023-04-01"]


    train_processor = DataProcessor(train, config_path)
    train_data = train_processor.get_processed_data()

    valid_processor = DataProcessor(valid, config_path)
    valid_data = valid_processor.get_processed_data()

    train_data.to_csv("statcast_2015-2023_cleaned.csv",index=False)
    valid_data.to_csv("statcast_2023-2024_cleaned.csv",index=False)



if __name__ == "__main__":

    config_path = 'config.json'
    data_path = pull_data()
    process_data(data_path,config_path)


    



