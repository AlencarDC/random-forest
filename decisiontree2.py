import math
from collections import Counter
from statistics import mode, mean
from typing import Union, List, Dict, Tuple
from enum import Enum

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

class DecisionTest:
  def __init__(self, value, column: int, expect=True):
    self.column = column
    self.value = value
    self.expect = expect

  def test(self, instance: List) -> bool:
    if len(instance)-1 < self.column:
      return False
    if isinstance(self.value, int) or isinstance(self.value, float):
      return (instance[self.column] >= self.value) == self.expect
    else:
      return (instance[self.column] == self.value) == self.expect

class DecisionLeaf:
  def __init__(self, predict: str, count: int):
    self.predict = predict
    self.count = count


class DecisionNode:
  def __init__(self, feature_name: str, gain: float):
    self.feature_name = feature_name
    self.gain = gain
    self.children: List[(DecisionTest, Union['DecisionNode', DecisionLeaf])] = []

  def add_child(self, test: DecisionTest, child: Union['DecisionNode', DecisionLeaf]):
    self.children.append((test, child))


class DecisionTree:
  def __init__(self):
    self.features: List[str] = []
    self._x: List[List] = []
    self._y: List = []
    self._root: Union[DecisionNode, DecisionLeaf] = None

  def build(self, x: List[List], y: List, features: List[str]):
    self._x = x
    self._y = y
    self.features = features

    self._root = self._build(self._x, self._y, list(self.features))

  def _build(self, x: List[List] , y: List, features: List):
    # All examples are of the same class
    if (entropy(y) == 0.0):
      return DecisionLeaf(y[0], len(y))

    # No more features
    if (len(x) == 0):
      return DecisionLeaf(most_frequent(y), len(y))

    # Get best feature to split
    best_col, best_gain = find_best_feature(x, y)

    # Give best feature to node
    node: DecisionNode = DecisionNode(features[best_col], best_gain)

    # Split data for the possible values of the best feature
    mean_value = 0 # Case numerical feature
    if isinstance(x[0][best_col], int) or isinstance(x[0][best_col], float):
      possible_values_data, mean_value = split_numerical(x, y, best_col)
    else:
      possible_values_data = split_categorical(x, y, best_col)

    # Remove feature from the possible features set
    feature_column = self.features.index(features[best_col])
    features.remove(features[best_col])

    # Create new nodes
    for value in possible_values_data:
      new_x, new_y = possible_values_data[value]
      if (len(new_x) == 0):
        #node.add_child(DecisionLeaf(most_frequent(new_y), len(new_y), value))
        return DecisionLeaf(most_frequent(new_y), len(new_y))
      else:
        # Check if represents numerical or categorical
        if isinstance(value, str):
          test = DecisionTest(value, feature_column)
        else:
          test = DecisionTest(mean_value, feature_column, value)
        node.add_child(test, self._build(new_x, new_y, list(features)))

    return node

  def predict(self, row: List):
    return self._predict(row, self._root)

  def _predict(self, row: List, node):
    if isinstance(node, DecisionLeaf):
      return node.predict

    for child in node.children:
      test, child_node = child
      if test.test(row) == True:
        return self._predict(row, child_node)

    return None





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
    label = node.predict + "\nCount: " + str(node.count)
  else:
    label = node.feature_name + "\nInfo Gain: " + str(node.gain)

  dot_node = "N"+ str(num) +" [label=\"" + label + "\"];\n"

  node_id = num
  if (isinstance(node, DecisionNode) == True):
    for idx, child in enumerate(node.children):
      if isinstance(child[0].value, int) or isinstance(child[0].value, float):
        operator = ">=" if child[0].expect == True else "<"
        edge_style = "[label=\""+ operator + str(child[0].value) +"\"]"
      else:
        edge_style = "[label=\""+ str(child[0].value) +"\"]"
      num += 1
      dot_node += "N" + str(node_id) + " -> N" + str(num) + edge_style + ";\n"
      dot, num = plot_node(child[1], num)
      dot_node += dot

  return dot_node, num



data = [
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
]


data2 = [
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
]

x = remove_column(data2, 4)
y = column(data2, 4)
tree = DecisionTree()
tree.build(x, y, ["Tempo","Temperatura","Umidade","Ventoso"])
plot_tree(tree)
print(tree.predict(["Chuvoso","Fria",150,"Falso"]))
