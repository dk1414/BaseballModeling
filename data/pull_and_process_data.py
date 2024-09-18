import pybaseball
import pandas as pd
import argparse
from datetime import datetime, timedelta
import sys
sys.path.append('../')
from DataProc.DataProcessor import DataProcessor


def pull_data(start_date, end_date):

    pybaseball.cache.enable()

    # Pull all pitch-level data for the specified date range
    full_data = pybaseball.statcast(start_dt=start_date, end_dt=end_date, verbose=True)

    # Add batter and pitcher names
    batter_names_df = pybaseball.playerid_reverse_lookup(full_data['batter'].to_list(), key_type='mlbam')
    pitcher_names_df = pybaseball.playerid_reverse_lookup(full_data['pitcher'].to_list(), key_type='mlbam')

    batter_names_df['full_name'] = batter_names_df['name_first'] + ' ' + batter_names_df['name_last']
    pitcher_names_df['full_name'] = pitcher_names_df['name_first'] + ' ' + pitcher_names_df['name_last']

    batter_names_dict = batter_names_df.set_index('key_mlbam')['full_name'].to_dict()
    pitcher_names_dict = pitcher_names_df.set_index('key_mlbam')['full_name'].to_dict()

    full_data['batter_name'] = full_data['batter'].map(batter_names_dict)
    full_data['pitcher_name'] = full_data['pitcher'].map(pitcher_names_dict)

    if 'player_name' in full_data.columns:
        full_data.drop(columns=['player_name'], inplace=True)

    deprecated_columns = ['spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'spin_dir', 'umpire']
    full_data.drop(columns=deprecated_columns, inplace=True)

    full_data.drop_duplicates(inplace=True)

    data_path = f'statcast_2015-2024.csv'
    full_data.to_csv(data_path, index=False)

    return data_path


def process_data(data_path, config_path, start_date, end_date):
    raw_data = pd.read_csv(data_path)

    # Convert game_date to datetime for easier comparison
    raw_data['game_date'] = pd.to_datetime(raw_data['game_date'])

    # Calculate 70-30 split date
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    split_date = pd.to_datetime(start_date) + timedelta(days=int(total_days * 0.7))

    # Split data into train and validation sets
    train = raw_data[raw_data['game_date'] < split_date]
    valid = raw_data[raw_data['game_date'] >= split_date]

    train_processor = DataProcessor(train, config_path)
    train_data = train_processor.get_processed_data()
    train_processor.save_scalers(f"statcast_{start_date}_{split_date.date()}_cleaned_scalers.pkl")

    valid_processor = DataProcessor(valid, config_path)
    valid_data = valid_processor.get_processed_data()
    valid_processor.save_scalers(f"statcast_{split_date.date()}_{end_date}_cleaned_scalers.pkl")

    train_data.to_csv(f"statcast_{start_date}_{split_date.date()}_cleaned.csv", index=False)
    valid_data.to_csv(f"statcast_{split_date.date()}_{end_date}_cleaned.csv", index=False)



if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Pull and process Statcast data.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for the data in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, required=True, help="End date for the data in YYYY-MM-DD format.")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the data file, optional.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    config_path = args.config_path

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = pull_data(start_date, end_date)

    #print("Processing Data")
    #process_data(data_path, config_path, start_date, end_date)


    



