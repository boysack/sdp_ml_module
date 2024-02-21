import pandas as pd

df = pd.read_csv("data/dataframe_interpolated_augmented.csv")

""" df['sensor_time'] = pd.to_datetime(df['sensor_time'])
df_sorted = df.sort_values(by='sensor_time').drop(columns=['Unnamed: 0'])
df_sorted.to_csv("dataframe_interpolated_augmented_sorted.csv", index=False) """
print(len(df.columns))
selected_columns = df[['ProdSpeed', 'LockDegree']]
print(len(selected_columns.columns))