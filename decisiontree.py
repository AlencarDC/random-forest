import numpy as np
from numpy import ndarray
from collections import Counter
from statistics import mode
from typing import Union

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

def entropy(rows: list) -> float:
  counter = Counter(rows)
  total = len(rows)

  result = 0
  for label in counter:
    result -= counter[label]/total  * np.log2(counter[label]/total)
  return result


def info_gain(rows: list, targets: list) -> float:
  total = len(rows)

  unique_values = set(rows)
  target_per_value = {val: [] for val in rows}

  for index, value in enumerate(rows):
    target_per_value[value].append(targets[index])

  result = 0
  for value in target_per_value:
    result += len(target_per_value[value])/total * entropy(target_per_value[value])

  return entropy(targets) - result


def find_best_feature(data: ndarray, target: list) -> int:
  gains = []

  num_features = len(data[0])
  for col in range(0, num_features):
    gains.append(info_gain(data[:, col], target))

  return gains.index(max(gains))

def most_frequent(target: list):
  return mode(target)

class DecisionLeaf:
  def __init__(self, value: str, count: int, feature: str):
    self.value = value
    self.count = count
    self.feature = feature

class DecisionNode:
  def __init__(self, col: int, label: str, feature: str):
    self.col = col
    self.label = label
    self.feature = feature
    self.children: list[Union['DecisionNode', DecisionLeaf]] = []

  def add_child(self, child: Union['DecisionNode', DecisionLeaf]):
    self.children.append(child)

class DecisionTree:
  def __init__(self):
    self.headers: list[str] = []
    self._x: ndarray = []
    self._y: list = []
    self._root: Union[DecisionNode, DecisionLeaf] = None

  def build(self, x: ndarray, y: list, headers: list[str]):
    self._x = x
    self._y = y
    self.headers = headers

    self._root = self._build(self._x, self._y, "")

  def _build(self, x , y, question):
    # All examples are of the same class
    if (entropy(y) == 0.0):
      return DecisionLeaf(y[0], len(y), question)

    # No more features
    if (len(x) == 0):
      return DecisionLeaf(most_frequent(y), len(y), question)

    # Get best feature to split
    best_col = find_best_feature(x, y)

    # Give best feature to node
    node: DecisionNode = DecisionNode(best_col, self.headers[best_col], question)

    # Split data for the possible values of the best feature
    possible_values = set(x[:, best_col])
    for value in possible_values:
      value_idxs = np.where(x[:, best_col] == value)
      new_x = x[value_idxs] # Best best feature row
      new_x = np.delete(new_x, best_col, 1) # Remove feature column
      new_y = y[value_idxs]
      if (len(data) == 0):
        #node.add_child(DecisionLeaf(most_frequent(new_y), len(new_y), value))
        return DecisionLeaf(most_frequent(new_y), len(new_y), value)
      else:
        node.add_child(self._build(new_x, new_y, value))

    return node



x = data[:,0:4]
y = data[:, 4]
tree = DecisionTree()
tree.build(x, y, ["Tempo", "Temperatura", "Umidade", "Ventoso"])
print(len(tree._root.children))
def print_tree(node, spacing):
  if (isinstance(node, DecisionNode) == True):
    print(spacing + "->" + node.label + "[" + node.feature + "]")
    for child in node.children:
      print_tree(child, spacing + "  ")

  if (isinstance(node, DecisionLeaf) == True):
    print(spacing + "+" + node.value + "[" + node.feature + "]")

print_tree(tree._root, "")
# result = entropy(data[:,4])
# print(result)

# for i in range(0,len(data[0])-1):
#   result = info_gain(data[:, i], data[:, 4])
#   print(result)

# best_col = find_best_feature(x, y)

# possible_values = set(x[:, best_col])
# for value in possible_values:
#   print(value)
#   value_idxs = np.where(x[:, best_col] == value)
#   new_x = x[value_idxs] # Best best feature row
#   new_x = np.delete(new_x, best_col, 1) # Remove feature column
#   new_y = y[value_idxs]

#   print(new_x)
#   print(new_y)
#   print(entropy(new_y))
#   print(info_gain(new_x, new_y))
