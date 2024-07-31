import pybaseball
import pandas as pd
import sys
sys.path.append('../')
from DataProc.DataProcessor import DataProcessor


def pull_data():

    pybaseball.cache.enable()

    #pull all pitch level data from 2015-2024
    full_data = pybaseball.statcast(start_dt="2015-01-01", end_dt="2024-06-30", verbose=True)



    #batter and pitcher columns are id's, lets add 2 columns to include their names
    #the player_name column already includes pitcher name, so we can just remove that one

    batter_names_df = pybaseball.playerid_reverse_lookup(full_data['batter'].to_list(), key_type='mlbam')
    pitcher_names_df = pybaseball.playerid_reverse_lookup(full_data['pitcher'].to_list(), key_type='mlbam')

    # Combine first and last names
    batter_names_df['full_name'] = batter_names_df['name_first'] + ' ' + batter_names_df['name_last']
    pitcher_names_df['full_name'] = pitcher_names_df['name_first'] + ' ' + pitcher_names_df['name_last']

    # Create dictionaries to map player IDs to full names
    batter_names_dict = batter_names_df.set_index('key_mlbam')['full_name'].to_dict()
    pitcher_names_dict = pitcher_names_df.set_index('key_mlbam')['full_name'].to_dict()

    # Map the full names to the full_data DataFrame
    full_data['batter_name'] = full_data['batter'].map(batter_names_dict)
    full_data['pitcher_name'] = full_data['pitcher'].map(pitcher_names_dict)


    if 'player_name' in full_data.columns:
        full_data.drop(columns=['player_name'], inplace=True)

    deprecated_columns = ['spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'spin_dir', 'umpire']
    full_data.drop(columns=deprecated_columns, inplace=True)

    #drop dups
    full_data.drop_duplicates(inplace=True)

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
    print("Processing Data")
    process_data(data_path,config_path)


    



