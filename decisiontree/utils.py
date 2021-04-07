import math
from collections import Counter
from statistics import mode, mean
from typing import Union, List, Dict, Tuple


def column(matrix, column) -> List:
  return [row[column] for row in matrix]

def remove_column(matrix, column) -> List:
  new_matrix = []
  for i, row in enumerate(matrix):
    new_matrix.append(list(row))
    del new_matrix[i][column]

  return new_matrix


def most_frequent(target: List):
  return mode(target)

def unique_values(my_list: List) -> List:
  return list(dict.fromkeys(my_list))

def entropy(rows: List) -> float:
  counter = Counter(rows)
  total = len(rows)

  result = 0
  for label in counter:
    result -= counter[label]/total  * math.log2(counter[label]/total)
  return result

# A possible optimization would be give the possbility to pass entropy(targets) as parameter
def info_gain_categorical(rows: List, targets: List) -> float:
  total = len(rows)

  # Initialize vectors of dict with features for each value of feature
  possible_values = unique_values(rows)
  target_per_value = {val: [] for val in possible_values}

  # Count ocurrences of class for each possible value
  for index, value in enumerate(rows):
    target_per_value[value].append(targets[index])

  result = 0
  for value in target_per_value:
    result += len(target_per_value[value])/total * entropy(target_per_value[value])

  return entropy(targets) - result

def info_gain_numerical(rows: List, targets: List) -> float:
  total = len(rows)

  mean_value = mean(rows)
  greater_count = 0
  greater_targets = []
  less_count = 0
  less_targets = []
  for idx, value in enumerate(rows):
    if value >= mean_value:
      greater_count += 1
      greater_targets.append(targets[idx])
    else:
      less_count += 1
      less_targets.append(targets[idx])

  result = greater_count / total * entropy(greater_targets)
  result += less_count / total * entropy(less_targets)

  return entropy(targets) - result

# Assumes that categorical features are string values and numerical features are int or float
def find_best_feature(data: List[List], target: List) -> (int, float):
  gains = []

  num_features = len(data[0])

  for col in range(0, num_features):
    if isinstance(data[0][col], int) or isinstance(data[0][col], float):
      gains.append(info_gain_numerical(column(data, col), target))
    else:
      gains.append(info_gain_categorical(column(data, col), target))
  print(gains)
  best_gain = max(gains)
  best_col = gains.index(best_gain)
  return best_col, best_gain

def split_categorical(old_x, old_y, col: int) -> Dict[str, Tuple[List, List]]:
  split = {}
  possible_values = unique_values(column(old_x, col))
  for value in possible_values:
    value_idxs = [idx for idx, row in enumerate(old_x) if row[col] == value] # Filtering
    new_x = []
    new_y = []
    for i, idx in enumerate(value_idxs):
      new_x.append(list(old_x[idx]))
      del new_x[i][col] # Delete feature column
      new_y.append(old_y[idx])

    split[value] = (new_x, new_y)

  return split

def split_numerical(old_x, old_y, col: int) -> (Dict[str, Tuple[List, List]], float):
    split = {}
    mean_value = mean(column(old_x, col))

    value_idxs = [idx for idx, row in enumerate(old_x) if row[col] >= mean_value] # Filtering
    new_x = []
    new_y = []
    for i, idx in enumerate(value_idxs):
      new_x.append(list(old_x[idx]))
      del new_x[i][col] # Delete feature column
      new_y.append(old_y[idx])
    split[True] = (new_x, new_y)

    value_idxs = [idx for idx, row in enumerate(old_x) if row[col] < mean_value] # Filtering
    new_x = []
    new_y = []
    for i, idx in enumerate(value_idxs):
      new_x.append(list(old_x[idx]))
      del new_x[i][col] # Delete feature column
      new_y.append(old_y[idx])
    split[False] = (new_x, new_y)

    return (split, mean_value)
