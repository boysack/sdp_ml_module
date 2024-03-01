import pandas as pd
import numpy as np

#TODO
#DROP: 2022-07-05 13:41:32 -> 2022-07-14 16:04:17 
#DROP: 2022-07-04 14:40:54 -> 2022-07-05 13:36:31 (INCLUDE EXTREMES)

def fill_and_interpolate():
    df = pd.read_csv("data/starting_point.csv", index_col=0)
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
    #gaussian_noise_interpolation(result_df)

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

def filter_datetime_range():
    # from date to date, in some cases there is too much delay
    df = pd.read_csv("data/linear_interpolated_all.csv", index_col=0)
    df['sensor_time'] = pd.to_datetime(df['sensor_time'])

    d01 = pd.Timestamp('2022-07-04 14:40:54')
    d02 = pd.Timestamp('2022-07-05 13:36:31')
    d03 = pd.Timestamp('2022-07-05 13:41:32')
    d04 = pd.Timestamp('2022-07-14 16:04:17')

    mask01 = (df['sensor_time'] <= d01)
    mask02 = (df['sensor_time'] >= d02) & (df['sensor_time'] <= d03)
    mask03 = (df['sensor_time'] >= d04)

    df[mask01].to_csv("data/linear_interpolated_part01.csv")
    df[mask02].to_csv("data/linear_interpolated_part02.csv")
    df[mask03].to_csv("data/linear_interpolated_part03.csv")

def fuse_parts():
    df01 = pd.read_csv("data/linear_interpolated_part01.csv", index_col=0)
    df02 = pd.read_csv("data/linear_interpolated_part02.csv", index_col=0)
    df03 = pd.read_csv("data/linear_interpolated_part03.csv", index_col=0)

    union_df = pd.concat([df01, df02, df03])

    union_df.to_csv("data/linear_interpolated_union.csv")

def save_mean_std():
    df01 = pd.read_csv("data/linear_interpolated_part01.csv", index_col=0).drop(columns=["sensor_time"])
    df02 = pd.read_csv("data/linear_interpolated_part02.csv", index_col=0).drop(columns=["sensor_time"])
    df03 = pd.read_csv("data/linear_interpolated_part03.csv", index_col=0).drop(columns=["sensor_time"])

    df01_means = df01.mean()
    df02_means = df02.mean()
    df03_means = df03.mean()

    df01_stds = df01.std()
    df02_stds = df02.std()
    df03_stds = df03.std()

    df_means = pd.concat([df01_means, df02_means, df03_means], axis=1)
    df_means = df_means.mean(axis=1)

    df_stds = pd.concat([df01_stds, df02_stds, df03_stds], axis=1)
    df_stds = df_stds.mean(axis=1)

    df_means.to_csv("data/metadata/df_original_means.csv")
    df_stds.to_csv("data/metadata/df_original_stds.csv")

if __name__=="__main__":
    #fill_and_interpolate()
    #filter_datetime_range()
    #fuse_parts()
    save_mean_std()