from kfold import kfold, FeatureType
from csv_handler import get_csv_data
import statistics

FILE_PATH = "vote.tsv"
CSV_SEPARATOR = "\t"
SEED = 42
K_FOLDS = 10
N_TREES = 3

data, features = get_csv_data(FILE_PATH, CSV_SEPARATOR)

feature_types = {
  "handicapped infants": FeatureType.CATEGORICAL,
  "water project cost sharing": FeatureType.CATEGORICAL,
  "adoption of the budget resolution": FeatureType.CATEGORICAL,
  "physician fee freeze": FeatureType.CATEGORICAL,
  "el salvador aid": FeatureType.CATEGORICAL,
  "religious groups in schools": FeatureType.CATEGORICAL,
  "anti satellite test ban": FeatureType.CATEGORICAL,
  "aid to nicaraguan contras": FeatureType.CATEGORICAL,
  "mx missile": FeatureType.CATEGORICAL,
  "immigration": FeatureType.CATEGORICAL,
  "synfuels corporation cutback": FeatureType.CATEGORICAL,
  "education spending": FeatureType.CATEGORICAL,
  "superfund right to sue": FeatureType.CATEGORICAL,
  "crime": FeatureType.CATEGORICAL,
  "duty free exports": FeatureType.CATEGORICAL,
  "export administration act south africa": FeatureType.CATEGORICAL
}

# Transform categorical features in strings
n_features = len(data[0])-1
for col in range(n_features):
  if (feature_types[features[col]] == FeatureType.CATEGORICAL):
    for instance in data:
      instance[col] = str(instance[col])

print("Accuracies:")
accuracies = kfold(K_FOLDS, data, features, n_trees=N_TREES, seed=SEED)
print(accuracies)
print(statistics.mean(accuracies))
print(statistics.stdev(accuracies))
