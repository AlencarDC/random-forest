import math
from collections import Counter
from typing import List
from .bootstrap import get_bootstrap
from .tree import DecisionTree
from .utils import remove_column, column

class RandomForest:
  def __init__(self, num_trees):
    self.num_trees = num_trees
    self.trees = []

  def train(self, x, y, features):
    m_attributes = int(math.sqrt(len(x[0])))

    for i in range(self.num_trees):
      tree = DecisionTree()
      training_set = get_bootstrap(x,y)
      tree.build(training_set[0], training_set[1], features, m_attributes)
      self.trees.append(tree)

  def predict(self, row: List):
    votes = []

    for tree in self.trees:
      new_vote = tree.predict(row)
      votes.append(new_vote)

    vote_count = Counter(votes)

    return vote_count.most_common(1)[0][0]


if __name__ == "__main__":
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

  x = remove_column(data, 4)
  y = column(4)
  features = ["Tempo","Temperatura","Umidade","Ventoso"]
  randomForest = RandomForest(1)
  randomForest.train(x,y,features)

  print("Expected: Nao | Actual:", randomForest.predict(["Ensolarado","Quente","Alta","Falso"]))
  print("Expected: Sim | Actual:", randomForest.predict(["Chuvoso","Amena","Alta","Falso"]))
