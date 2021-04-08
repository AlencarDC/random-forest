from typing import List
import random
from randomforest import forest
from randomforest.utils import remove_column, column

features = ["Index", "Tempo","Temperatura","Umidade","Ventoso"]
data = [
  [1,"Ensolarado","Quente","Alta","Falso","Nao"],
  [2, "Ensolarado","Quente","Alta","Verdadeiro","Nao"],
  [3,"Nublado","Quente","Alta","Falso","Sim"],
  [4,"Chuvoso","Amena","Alta","Falso","Sim"],
  [5,"Chuvoso","Fria","Normal","Falso","Sim"],
  [6,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [7,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [8,"Ensolarado","Amena","Alta","Falso","Nao"],
  [9,"Ensolarado","Fria","Normal","Falso","Sim"],
  [10,"Chuvoso","Amena","Normal","Falso","Sim"],
  [11,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [12,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [13,"Nublado","Quente","Normal","Falso","Sim"],
  [14,"Chuvoso","Fria","Normal","Falso","Sim"],
  [15,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [16,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [17,"Ensolarado","Amena","Alta","Falso","Nao"],
  [18,"Ensolarado","Fria","Normal","Falso","Sim"],
  [19,"Chuvoso","Amena","Normal","Falso","Sim"],
  [20,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [21,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [22,"Nublado","Quente","Normal","Falso","Sim"],
  [23,"Chuvoso","Fria","Normal","Falso","Sim"],
  [24,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [25,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [26,"Ensolarado","Amena","Alta","Falso","Nao"],
  [27,"Ensolarado","Fria","Normal","Falso","Sim"],
  [28,"Chuvoso","Amena","Normal","Falso","Sim"],
  [29,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [30,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [31,"Nublado","Quente","Normal","Falso","Sim"],
  [32,"Chuvoso","Amena","Alta","Verdadeiro","Nao"]
]

def get_kfolds(k: int, data: List, shuffle=False, seed=42):
  if shuffle == True:
    random.seed(42)
    random.shuffle(data)

  fold_size = int(len(data) / k)

  folds = []

  first_fold_size = fold_size+len(data) % k
  folds.append(data[0:first_fold_size])
  print("Data Size: " + str(len(data)))
  print("Fold size: " + str(fold_size))
  print("First fold size: " + str(first_fold_size))
  for i in range(k-1):
    begin = first_fold_size + i * fold_size
    end = begin + fold_size
    folds.append(data[begin:end])

  return folds


n_trees = 40
k = 10

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

print(results)





