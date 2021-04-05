import numpy as np
from numpy import ndarray
from collections import Counter
from statistics import mode
from typing import Union, List

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

def entropy(rows: List) -> float:
  counter = Counter(rows)
  total = len(rows)

  result = 0
  for label in counter:
    result -= counter[label]/total  * np.log2(counter[label]/total)
  return result

def unique_values(my_list: List) -> List:
  return list(dict.fromkeys(my_list))

def info_gain(rows: List, targets: List) -> float:
  total = len(rows)

  #unique_values = unique_values(rows)
  target_per_value = {val: [] for val in rows}

  for index, value in enumerate(rows):
    target_per_value[value].append(targets[index])

  result = 0
  for value in target_per_value:
    result += len(target_per_value[value])/total * entropy(target_per_value[value])

  return entropy(targets) - result


def find_best_feature(data: ndarray, target: List) -> int:
  gains = []

  num_features = len(data[0])
  for col in range(0, num_features):
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

class DecisionTree:
  def __init__(self):
    self.headers: List[str] = []
    self._x: ndarray = []
    self._y: List = []
    self._root: Union[DecisionNode, DecisionLeaf] = None

  def build(self, x: ndarray, y: List, headers: List[str]):
    self._x = x
    self._y = y
    self.headers = headers

    self._root = self._build(self._x, self._y, list(headers), "")

  def _build(self, x , y, headers: List, question):
    # All examples are of the same class
    if (entropy(y) == 0.0):
      return DecisionLeaf(y[0], len(y), question)

    # No more features
    if (len(x) == 0):
      return DecisionLeaf(most_frequent(y), len(y), question)

    # Get best feature to split
    print("\n\n"+question)
    print(x)
    best_col, best_gain = find_best_feature(x, y)
    print(best_col, headers)
    #print("Best feature " + str(best_col) + ":" + str(headers[best_col]) + " with " + str(best_gain))

    # Give best feature to node
    node: DecisionNode = DecisionNode(best_col, headers[best_col], best_gain, question)
    headers.remove(headers[best_col])

    # Split data for the possible values of the best feature
    possible_values = unique_values(x[:, best_col])
    for value in possible_values:
      value_idxs = np.where(x[:, best_col] == value)
      new_x = x[value_idxs] # Best best feature row
      new_x = np.delete(new_x, best_col, axis=1) # Remove feature column
      new_y = y[value_idxs]
      if (len(data) == 0):
        #node.add_child(DecisionLeaf(most_frequent(new_y), len(new_y), value))
        return DecisionLeaf(most_frequent(new_y), len(new_y), value)
      else:
        node.add_child(self._build(new_x, new_y, list(headers), value))

    return node



x = data[:,0:4]
y = data[:, 4]
tree = DecisionTree()
tree.build(x, y, ["Tempo", "Temperatura", "Umidade", "Ventoso"])
print(len(tree._root.children))


def print_tree(node, spacing):
  if (isinstance(node, DecisionNode) == True):
    print(spacing + "->" + node.label + "[" + node.feature + "] with " + str(node.gain))
    for child in node.children:
      print_tree(child, spacing + "  ")

  if (isinstance(node, DecisionLeaf) == True):
    print(spacing + "+" + node.value + "[" + node.feature + "]")

print_tree(tree._root, "")
