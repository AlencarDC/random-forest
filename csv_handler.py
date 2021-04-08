import pandas
from pathlib import Path

def get_csv_data(filepath, separator=","):
  file_data = pandas.read_csv(filepath, sep=separator)

  # Get a list of instances (lists of attributes)
  instances = file_data.values.tolist()
  headers = file_data.columns.values.tolist()

  return (instances, headers)

def save_csv_file(filepath, data_set):
  output_path = Path(filepath)
  output_dir = output_path.parent
  output_dir.mkdir(parents=True, exist_ok=True)

  dataframe = pandas.DataFrame(data_set)
  dataframe.to_csv(output_path, index=False)
