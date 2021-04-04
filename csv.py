import pandas
from pathlib import Path

def get_csv_data(filepath, target_col="target", separator=","):
  file_data = pandas.read_csv(filepath, sep=separator)

  # Get a list of instances (lists of attributes)
  only_attrs = file_data.drop([target_col], axis=1)
  attributes = only_attrs.values
  headers = only_attrs.columns.values

  # Get list of the expected values for the attributes classification
  targets = [target[0] for target in file_data[[target_col]].values]

  return (attributes, targets, headers)

def save_csv_file(filepath, data_set):
  output_path = Path(filepath)
  output_dir = output_path.parent
  output_dir.mkdir(parents=True, exist_ok=True)

  dataframe = pandas.DataFrame(data_set)
  dataframe.to_csv(output_path, index=False)
