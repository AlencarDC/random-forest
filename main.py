from kfold import kfold
from csv_handler import get_csv_data

data, features = get_csv_data("vote.tsv", "\t")

# Transform categorical features in strings
n_features = len(data[0])-1
for instance in data:
  for col in range(n_features):
    instance[col] = str(instance[col])

print("Accuracies:")
print(kfold(10, data, features, n_trees=1))
