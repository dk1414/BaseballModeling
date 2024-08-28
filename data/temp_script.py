import pandas as pd
from datetime import datetime, timedelta




def main():

    t = pd.read_csv("statcast_2015-04-01_2021-04-04_cleaned.csv")
    raw_data = t
    start_date = pd.to_datetime("2017-01-01")
    end_date = pd.to_datetime("2018-01-01")
    raw_data['game_date'] = pd.to_datetime(raw_data['game_date'])
    total_days = (end_date - start_date).days
    split_date = pd.to_datetime(start_date) + timedelta(days=int(total_days * 0.6))

    # Split data into train and validation sets
    train = raw_data[(raw_data['game_date'] > start_date) & (raw_data['game_date'] < split_date)]
    valid = raw_data[(raw_data['game_date'] > split_date) & (raw_data['game_date'] < end_date)]
    train.to_csv(f"med_train.csv", index=False)
    valid.to_csv(f"med_test.csv", index=False)


if __name__ == "__main__":
    main()
