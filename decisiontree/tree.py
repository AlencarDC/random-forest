from typing import Union, List, Dict, Tuple
from enum import Enum
import numpy as np
from utils import entropy, find_best_feature, most_frequent
from utils import split_categorical, split_numerical

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


########### PLOT TREE ############

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

if __name__ == "__main__":
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
  features = ["Tempo","Temperatura","Umidade","Ventoso"]
  x = data[:,0:4]
  y = data[:, 4]

  tree = DecisionTree()
  tree.build(x, y, features)
  plot_tree(tree)
