from typing import List
import random
from randomforest import forest
from randomforest.utils import remove_column, column

def get_kfolds(k: int, data: List, shuffle=False, seed=42) -> List:
  if shuffle == True:
    random.seed(42)
    random.shuffle(data)

  fold_size = int(len(data) / k)

  folds = []

  first_fold_size = fold_size+len(data) % k
  folds.append(data[0:first_fold_size])

  for i in range(k-1):
    begin = first_fold_size + i * fold_size
    end = begin + fold_size
    folds.append(data[begin:end])

  return folds


def kfold(k: int, data: List, features: List, n_trees=41) -> List:
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







