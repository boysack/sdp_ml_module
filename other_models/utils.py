import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from matplotlib import pyplot as plt


def col_v(vec):
  vec = np.array(vec)
  return vec.reshape(len(vec), 1)

def generate_samples_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
  data = pd.read_csv("./data/falsa_interpolazione_fatta_da_me.csv") # load the data into a pandas dataframe
  data = data.drop(data.columns[0], axis=1) # drop the first column (index column)

  # Convert the 'sensor_time' column to datetime
  data['sensor_time'] = pd.to_datetime(data['sensor_time'])

  # Create a new column for the next second
  data['next_second'] = data['sensor_time'] + pd.Timedelta(seconds=1)

  # Merge the data with itself on the 'sensor_time' and 'next_second' columns
  merged = pd.merge(data, data, left_on='sensor_time', right_on='next_second', suffixes=('_current', '_next'))

  # The 'samples' are the current rows and the 'labels' are the next rows
  samples = merged.filter(regex='_current$')
  labels = merged.filter(regex='_next$')

  samples = samples.rename(columns=lambda x: x.replace('_current', ''))
  labels = labels.rename(columns=lambda x: x.replace('_next', ''))


  samples = samples.drop('next_second', axis=1)
  labels = labels.drop('next_second', axis=1)

  scalers = {}

  # Normalize each column separately
  for column in samples.columns:
    if column != 'sensor_time':
      scaler = StandardScaler()
      samples[column] = scaler.fit_transform(samples[[column]])
      labels[column] = scaler.transform(labels[[column]])
      scalers[column] = scaler

  # Save the scalers
  joblib.dump(scalers, 'scalers.pkl')

  return samples, labels

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self._example = None

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width # The number of time steps included in the input
    self.label_width = label_width # The number of time steps included in the label
    self.shift = shift # The number of time steps to predict into the future

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='Tcpu', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=4,)
  

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result
@example.setter
def example(self, _):
    self._example = _

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


