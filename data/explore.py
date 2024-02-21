import pandas as pd
import numpy as np

#TODO: 2022-07-05 13:41:32 -> 2022-07-14 16:04:17 => DROP (too large interpolation)

def fill_and_interpolate():
    df = pd.read_csv("data/starting_point.csv")
    # Convert 'sensor_time' column to datetime type
    df['sensor_time'] = pd.to_datetime(df['sensor_time'])

    # Sort by 'sensor_time'
    df = df.sort_values(by="sensor_time")

    # Create a new DataFrame with all seconds between the min and max sensor_time
    all_seconds = pd.date_range(start=df['sensor_time'].min(), end=df['sensor_time'].max(), freq='S')
    complete_df = pd.DataFrame({'sensor_time': all_seconds})

    # Merge the original DataFrame with the complete DataFrame to fill missing rows
    result_df = pd.merge(complete_df, df, on='sensor_time', how='left')

    linear_interpolation(result_df)
    gaussian_noise_interpolation(result_df)

def linear_interpolation(df):
    # Interpolate missing values
    df = df.interpolate(method='linear')
    df.to_csv("data/linear_interpolated_all.csv")

def gaussian_noise_interpolation(df):
    for col in df.columns:
        missing_indices = df[col].index[df[col].isnull()].tolist()
        for idx in missing_indices:
            mean = df[col].mean()
            std_dev = df[col].std()
            noise = np.random.normal(mean, std_dev)
            df.at[idx, col] = noise
    
    df.to_csv("data/gaussian_interpolated_all.csv")

if __name__=="__main__":
    fill_and_interpolate()