import pandas as pd
import re
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

expanded_data = []

def merge_csv_files(file1, file2, output_file):
    # Carica i due file CSV
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatena i due DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True, sort=False)

    # Salva il DataFrame unito in un nuovo file CSV
    merged_df.to_csv(output_file, index=False)

    print(f"File merged and saved as '{output_file}'")

def convert_to_seconds(time_str):
    # Dividi la stringa in giorni, ore, minuti e secondi
    parts = time_str.split(':')
    
    # Estrai i numeri da ciascuna parte
    days, hours, minutes, seconds = [int(part[:-1]) for part in parts]

    # Converti tutto in secondi e somma
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds

    return total_seconds

# Test della funzione
def load_and_clean_data(file_path):
    # Carica i dati
    df = pd.read_csv(file_path)

    ex_sensor = ['TotalProduct', 'stsTotalCount', 'RunningTime', 'PowerOnTime', 'Index', 'OperationMode', 'OperationState', 'Alarm', ]

    # Espandi l'array 'samples' in righe separate
    df = df.explode('samples')
    df = df.dropna(subset=['first_time', 'last_time'])

    # Assicurati che ogni elemento in 'samples' sia un dizionario
    df['samples'] = df['samples'].apply(lambda x: eval(x) if isinstance(x, str) else x) #check??

    # Crea un nuovo dataframe per i dati espansi
    for index, row in df.iterrows():
        for sample in row['samples']:
            if sample and 'name' in sample and 'value' in sample and 'time' in sample and sample['name'] not in ex_sensor:
                name = sample['name'].split('_')[-1]
                if name not in ex_sensor:
                    expanded_data.append({
                        'type': row['folder'],
                        '_id': row['_id'],
                        'sensor_name': name, #sample['name'],
                        'sensor_value': sample['value'],
                        'sensor_time': pd.to_datetime(sample['time'], unit='ms')
                    })
            elif sample and row['variable'] == 'test':
                pass
            elif sample and 'value' in sample and 'time' in sample and row['variable'] and not re.match(r'test[1-3]', row['variable']) and row['variable'] not in ex_sensor: 
                if isinstance(sample['value'], str):
                        val = convert_to_seconds(sample['value']) #??
                        expanded_data.append({
                            'type': 'PLC',
                            '_id': row['_id'],
                            'sensor_name': row['variable'],
                            'sensor_value': val,
                            'sensor_time': pd.to_datetime(sample['time'], unit='ms')
                        })
                else:
                    expanded_data.append({
                        'type': 'PLC',
                        '_id': row['_id'],
                        'sensor_name': row['variable'],
                        'sensor_value': sample['value'],
                        'sensor_time': pd.to_datetime(sample['time'], unit='ms')
                })

    expanded_df = pd.DataFrame(expanded_data)
    
    # Riordina le colonne se necessario
    expanded_df = expanded_df[['sensor_time', 'sensor_name', 'sensor_value']]
    expanded_df['sensor_time'] = pd.to_datetime(expanded_df['sensor_time'])
    expanded_df['sensor_time'] = expanded_df['sensor_time'].dt.round('s')
    # Pivot dei dati con sensor_time arrotondato e index come indici
    pivoted_df = expanded_df.pivot_table(index=['sensor_time'], columns='sensor_name', values='sensor_value', aggfunc='mean')
    pivoted_df.reset_index(inplace=True)

    return pivoted_df

def plot_correlation(D):
    pearson_correlations(D, "dataset")

def pearson_correlations(D, Title, cmap="Purples"):

    df = D.loc[:, D.var() != 0]
    corr_matrix = df.corr()
    corr_matrix.to_csv('./plot_correlations/corr_matrix.csv')
    print(corr_matrix)
    sns.set()
    heatmap = sns.heatmap(
        corr_matrix,
        linewidth=0.2,
        cmap=cmap,
        square=True,
        cbar=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title(Title)
    plt.savefig('./plot_correlations/' + 'heatmap_%s.pdf' % (Title))
    plt.show()
    #plt.close()

    return

# Percorsi dei file sorgente
file1 = 'data/arol.csv'
file2 = 'data/arol_drive.csv'
file3 = 'data/arol_plc.csv'

# Percorso del file di output
output_file_1 = 'data/merged_arol_1.csv'
output_file_2 = 'data/merged_arol_final.csv'

# Esegui la funzione di unione
merge_csv_files(file1, file2, output_file_1)
merge_csv_files(output_file_1, file3, output_file_2)
# Carica il dataset
file_path = output_file_2

df = load_and_clean_data(file_path)
df.to_csv("data/cleaned_data.csv")

df['sensor_time'] = pd.to_datetime(df['sensor_time'])

# Seleziona solo le colonne numeriche
numeric_cols = df.select_dtypes(include=[np.number]).columns

zero_sensors = [] # sensors with only value 0
fixed_sensors = [] # sensors with only the same value

min_values = df[numeric_cols].min() # min value for each sensor to do zero check and fixed value check
max_values = df[numeric_cols].max() # max value for each sensor to do zero check and fixed value check

equal_values_cols = numeric_cols[min_values == max_values] # sensors with only the same value
zero_sensors = equal_values_cols[df[equal_values_cols].iloc[0] == 0] # sensors with only value 0
fixed_sensors = equal_values_cols[df[equal_values_cols].iloc[0] != 0] # sensors with only the same value

df.drop(zero_sensors, axis=1, inplace=True) # drop sensors with only value 0
df.drop(fixed_sensors, axis=1, inplace=True) # drop sensors with only the same value

numeric_cols = df.select_dtypes(include=[np.number]).columns # update numeric_cols

# Per ogni colonna numerica
for col in numeric_cols:
    # Se il primo elemento è NaN, sostituiscilo con la moda dei valori successivi
    if pd.isna(df[col].iloc[0]): # perché a 0?
        mode_value = df[col][1:].mode()[0]  # Prendi la moda dei valori successivi
        df.loc[df.index[0], col] = mode_value

df_column_reduced = df.copy()

noTcolumns = df_column_reduced.columns[(~df.columns.str.startswith('T')) & (df.columns != "sensor_time")] # sensors without 'T' in the name
#df_column_reduced.drop(noTcolumns, axis=1, inplace=True) # drop sensors with 'T' in the name

############################# Data augmentation #############################

df_initial_interpolation = df_column_reduced.copy()

delays = [] # list of delays between rows

df.index = range(len(df))
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate(method='polynomial', order=2)

i = 0
concatenati = set()
df_array = []
while i <= len(df) - 1: # iterate through the length of the dataframe, we use a while because the size of the dataframe grows
    actual_row = df.iloc[i] # get the current row
    if i != len(df)-1:
        next_row = df.iloc[i+1] # get the next row

    actual_row_time = actual_row['sensor_time'] # get the timestamp of the current row
    next_row_time = next_row['sensor_time'] # get the timestamp of the next row

    delay_between_rows = (next_row_time - actual_row_time).seconds # calculate the delay between the two rows

    if delay_between_rows > 1 or i == len(df)-1: # if the delay is greater than 1 second and less than 10 minutes grazie costa <3
        concatenati.add(actual_row_time)

        df1 = df.iloc[:i+1] # slice of the dataframe, from the beginning to the current row
        df_array.append(df1)
        df = df.iloc[i+1:] # slice of the dataframe, from the next row to the end
        i = 0
    else:
        i += 1 # increment the counter

# Inizializza una lista vuota per le nuove finestre
new_windows = []


# Definisci le colonne su cui vuoi basare la distribuzione
columns = df[df.columns.difference(['sensor_time'])].columns

# Crea una finestra di riferimento che rappresenti la distribuzione media di tutte le finestre
reference_window = pd.concat(df_array)[columns]


# Itera su ogni DataFrame in df_array
for i in range(len(df_array) - 1):
    df = df_array[i]
    next_df = df_array[i + 1]

    # Calcola la differenza di tempo in secondi tra l'ultimo elemento della finestra corrente e il primo elemento della finestra successiva
    time_diff = (next_df['sensor_time'].iloc[0] - df['sensor_time'].iloc[-1]).total_seconds()
    n = min(time_diff,100)

    # Inizializza un nuovo DataFrame vuoto per la finestra
    new_window = pd.DataFrame()

    # Crea una serie di timestamp per la colonna sensor_time
    start_time = df['sensor_time'].iloc[-1]
    end_time = start_time + pd.Timedelta(seconds=n) - pd.Timedelta(seconds=1)
    new_window['sensor_time'] = pd.date_range(start=start_time, end=end_time, freq="S")[1:]

    # Itera su ogni colonna
    for column in columns:
        distribution = reference_window[column].value_counts() # Calcola la distribuzione della colonna
        # Crea una nuova serie di dati con la stessa distribuzione
        series_to_sample = df[df[column].isin(distribution.index)][column]
        if not series_to_sample.empty:  # Check if the series is not empty
            new_series = series_to_sample.sample(n=int(n), replace=True).reset_index(drop=True)
            # Aggiungi la nuova serie al nuovo DataFrame
            new_window[column] = new_series
    
    # Aggiungi la nuova finestra alla lista di nuove finestre
    new_windows.append(new_window)



df = pd.DataFrame()  # Initialize df as an empty DataFrame if it's not already defined

i = 0
for new_window in new_windows:
    temp_df = pd.concat([df_array[i], new_window])  # Concatenate df_array[i] and new_window
    plot_features = temp_df[:]
    df = pd.concat([df, temp_df])  # Append the result to df
    i += 1
#df.interpolate(method='linear',inplace=True)
#df.dropna(inplace=True)

noTcolumns = df.columns[(~df.columns.str.startswith('T')) & (df.columns != "sensor_time")] # sensors without 'T' in the name
#df.drop(noTcolumns, axis=1, inplace=True) # drop sensors with 'T' in the name

df.dropna(inplace=True)

df = df.iloc[:len(df)//31*31]

# Split df into blocks of 31 rows
blocks = [df.iloc[i:i+31] for i in range(0, len(df), 31)]

# Shuffle blocks
np.random.shuffle(blocks)

# Concatenate shuffled blocks back into a single DataFrame
df = pd.concat(blocks, ignore_index=True)
df.to_csv("data/dataframe_interpolated_augmented.csv")
#############################################################################