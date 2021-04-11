from typing import List
from enum import Enum
import random
from randomforest import forest
from randomforest.utils import remove_column, column, unique_values

class FeatureType(Enum):
  CATEGORICAL = 0
  NUMERICAL = 1

def get_kfolds(k: int, data: List, shuffle=False) -> List:
  if shuffle == True:
    random.shuffle(data)

  n_columns = len(data[0])
  # Get possible values for target column
  possible_values = unique_values(column(data, n_columns-1))

  # Initialize dict of index of instances per target value
  stratified_data = {key: [] for key in possible_values}

  # Divide by indexes of data per target value
  for idx, instance in enumerate(data):
    stratified_data[instance[n_columns-1]].append(idx)

  # Initialize folds
  folds = [[] for i in range(k)]

  # Create folds
  for key in stratified_data:
    for i, idx in enumerate(stratified_data[key]):
      folds[i % k].append(list(data[idx]))

  return folds


def kfold(k: int, data: List, features: List, n_trees=41, seed=42) -> List:
  folds = get_kfolds(k, data)
  results = []

  for i in range(k):
    testing_fold = folds[i]

    # Union of all fold except testing_fold
    training_folds = []
    for j in range(k):
      if j != i:
        training_folds.extend(folds[j])

    target_column = len(training_folds[0]) - 1
    x = remove_column(training_folds, target_column)
    y = column(training_folds, target_column)

    # Create Random Forest model
    rf = forest.RandomForest(n_trees)
    rf.train(x, y, features)

    # Run prediction over testing fold
    x_testing = remove_column(training_folds, target_column)
    y_testing = column(training_folds, target_column)

    total_predictions = len(training_folds)
    right_preditions = 0
    for j in range(total_predictions):
      predicted = rf.predict(x_testing[j])
      if predicted == y_testing[j]:
        right_preditions += 1

    accuracy = right_preditions / total_predictions
    results.append(accuracy)

  return results







