import numpy as np
from numpy import ndarray
from collections import Counter
from statistics import mode, mean
from typing import Union, List, Dict, Tuple
from enum import Enum

data = np.array([
  ["Ensolarado","Quente","Alta","Falso","Nao"],
  ["Ensolarado","Quente","Alta","Verdadeiro","Nao"],
  ["Nublado","Quente","Alta","Falso","Sim"],
  ["Chuvoso","Amena","Alta","Falso","Sim"],
  ["Chuvoso","Fria","Normal","Falso","Sim"],
  ["Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  ["Nublado","Fria","Normal","Verdadeiro","Sim"],
  ["Ensolarado","Amena","Alta","Falso","Nao"],
  ["Ensolarado","Fria","Normal","Falso","Sim"],
  ["Chuvoso","Amena","Normal","Falso","Sim"],
  ["Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  ["Nublado","Amena","Alta","Verdadeiro","Sim"],
  ["Nublado","Quente","Normal","Falso","Sim"],
  ["Chuvoso","Amena","Alta","Verdadeiro","Nao"]
])

data2 = np.array([
  ["Ensolarado","Quente",110,"Falso","Nao"],
  ["Ensolarado","Quente",110,"Verdadeiro","Nao"],
  ["Nublado","Quente",110,"Falso","Sim"],
  ["Chuvoso","Amena",105,"Falso","Sim"],
  ["Chuvoso","Fria",90,"Falso","Sim"],
  ["Chuvoso","Fria",90,"Verdadeiro","Nao"],
  ["Nublado","Fria",95,"Verdadeiro","Sim"],
  ["Ensolarado","Amena",110,"Falso","Nao"],
  ["Ensolarado","Fria",90,"Falso","Sim"],
  ["Chuvoso","Amena",90,"Falso","Sim"],
  ["Ensolarado","Amena",90,"Verdadeiro","Sim"],
  ["Nublado","Amena",110,"Verdadeiro","Sim"],
  ["Nublado","Quente",90,"Falso","Sim"],
  ["Chuvoso","Amena",110,"Verdadeiro","Nao"]
])

def entropy(rows: List) -> float:
  counter = Counter(rows)
  total = len(rows)

  result = 0
  for label in counter:
    result -= counter[label]/total  * np.log2(counter[label]/total)
  return result

def unique_values(my_list: List) -> List:
  return list(dict.fromkeys(my_list))

# A possible optimization would be give the possbility to pass entropy(targets) as parameter
def info_gain(rows: List, targets: List) -> float:
  total = len(rows)

  # Initialize vectors of dict with features for each value of feature
  possible_values = unique_values(rows)
  target_per_value = {val: [] for val in possible_values}

  for index, value in enumerate(rows):
    target_per_value[value].append(targets[index])

  result = 0
  for value in target_per_value:
    result += len(target_per_value[value])/total * entropy(target_per_value[value])

  return entropy(targets) - result

# A possible optimization would be give the possbility to pass entropy(targets) as parameter
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
def find_best_feature(data: ndarray, target: List) -> (int, float):
  gains = []

  num_features = len(data[0])
  for col in range(0, num_features):
    if np.char.isnumeric(data[0, col]):
      gains.append(info_gain_numerical(data[:, col].astype(float), target))
    else:
      gains.append(info_gain(data[:, col], target))
  print(gains)
  best_gain = max(gains)
  return gains.index(best_gain), best_gain

def most_frequent(target: List):
  return mode(target)

class DecisionLeaf:
  def __init__(self, value: str, count: int, feature: str):
    self.value = value
    self.count = count
    self.feature = feature

class DecisionNode:
  def __init__(self, col: int, label: str, gain: float, feature: str):
    self.col = col
    self.label = label
    self.gain = gain
    self.feature = feature
    self.children: List[Union['DecisionNode', DecisionLeaf]] = []

  def add_child(self, child: Union['DecisionNode', DecisionLeaf]):
    self.children.append(child)

class FeatureType(Enum):
  NUMERICAL = 1
  CATEGORICAL = 2

class DecisionTree:
  def __init__(self):
    self.features: List[str] = []
    self._x: ndarray = []
    self._y: List = []
    self._root: Union[DecisionNode, DecisionLeaf] = None

  def build(self, x: ndarray, y: List, features_type: Dict[str, FeatureType]):
    self._x = x
    self._y = y
    self.features = list(features_type.keys())
    self.features_type = features_type

    self._root = self._build(self._x, self._y, list(self.features), "<root>")

  def _build(self, x , y, features: List, question):
    # All examples are of the same class
    if (entropy(y) == 0.0):
      return DecisionLeaf(y[0], len(y), question)

    # No more features
    if (len(x) == 0):
      return DecisionLeaf(most_frequent(y), len(y), question)

    # Get best feature to split
    best_col, best_gain = find_best_feature(x, y)

    # Give best feature to node
    node: DecisionNode = DecisionNode(best_col, features[best_col], best_gain, question)

    # Split data for the possible values of the best feature
    if self.features_type[features[best_col]] == FeatureType.CATEGORICAL:
      possible_values_data = self.split_categorical(x, y, best_col)
    else:
      possible_values_data = self.split_numerical(x, y, best_col)

    # Remove feature from the possible features set
    features.remove(features[best_col])

    # Create new nodes
    for value in possible_values_data:
      new_x, new_y = possible_values_data[value]
      if (len(new_x) == 0):
        #node.add_child(DecisionLeaf(most_frequent(new_y), len(new_y), value))
        return DecisionLeaf(most_frequent(new_y), len(new_y), value)
      else:
        node.add_child(self._build(new_x, new_y, list(features), value))

    return node

  def split_categorical(self, old_x, old_y, col: int) -> Dict[str, Tuple[List, List]]:
    split = {}
    possible_values = unique_values(old_x[:, col])
    for value in possible_values:
      value_idxs = np.where(old_x[:, col] == value)
      new_x = old_x[value_idxs] # Best best feature row
      new_x = np.delete(new_x, col, axis=1) # Remove feature column
      new_y = old_y[value_idxs]
      split[value] = (new_x, new_y)

    return split

  def split_numerical(self, old_x, old_y, col: int) -> Dict[str, Tuple[List, List]]:
    split = {}
    mean_value = mean(old_x[:, col].astype(float))

    value_idxs = np.where(old_x[:, col].astype(float) >= mean_value)
    new_x = old_x[value_idxs] # Best best feature row
    new_x = np.delete(new_x, col, axis=1) # Remove feature column
    new_y = old_y[value_idxs]
    split[">=" + str(mean_value)] = (new_x, new_y)

    value_idxs = np.where(old_x[:, col].astype(float) < mean_value)
    new_x = old_x[value_idxs] # Best best feature row
    new_x = np.delete(new_x, col, axis=1) # Remove feature column
    new_y = old_y[value_idxs]
    split["<" + str(mean_value)] = (new_x, new_y)

    return split

x = data2[:,0:4]
y = data2[:, 4]
tree = DecisionTree()
features_and_types = {
  "Tempo": FeatureType.CATEGORICAL,
  "Temperatura": FeatureType.CATEGORICAL,
  "Umidade": FeatureType.NUMERICAL,
  "Ventoso": FeatureType.CATEGORICAL
}
tree.build(x, y, features_and_types)
print(len(tree._root.children))


def print_tree(node, spacing):
  if (isinstance(node, DecisionNode) == True):
    print(spacing + "->" + node.label + "[" + node.feature + "] with " + str(node.gain))
    for child in node.children:
      print_tree(child, spacing + "  ")

  if (isinstance(node, DecisionLeaf) == True):
    print(spacing + "+" + node.value + "[" + node.feature + "]")

def plot_tree(tree: DecisionTree):
  graph = "digraph G {\n"
  graph += "node [shape=box,style=bold]\n"
  graph += "edge [fontsize=10]\n"
  new_nodes, _ = plot_node(tree._root, 0)
  graph += new_nodes
  graph += "}"
  print(graph)

def plot_node(node: Union[DecisionNode, DecisionLeaf], num: int) -> (str, int):
  if (isinstance(node, DecisionLeaf) == True):
    label = node.value + "\nCount: " + str(node.count)
  else:
    label = node.label + "\nInfo Gain: " + str(node.gain)

  dot_node = "N"+ str(num) +" [label=\"" + label + "\"];\n"

  node_id = num
  if (isinstance(node, DecisionNode) == True):
    for idx, child in enumerate(node.children):
      edge_style = "[label=\""+ child.feature +"\"]"
      num += 1
      dot_node += "N" + str(node_id) + " -> N" + str(num) + edge_style + ";\n"
      dot, num = plot_node(child, num)
      dot_node += dot

  return dot_node, num


#print_tree(tree._root, "")
plot_tree(tree)

