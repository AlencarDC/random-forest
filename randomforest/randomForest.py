from bootstrap import get_bootstrap
from tree import DecisionTree
import math
import numpy as np
from numpy import ndarray

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

x = data[:,0:4]
y = data[:, 4]
features = ["Tempo","Temperatura","Umidade","Ventoso"]

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




if __name__ == "__main__":
    randomForest = RandomForest(2)
    randomForest.train(x,y,features)

    randomForest.trees[0].plot_tree()
    randomForest.trees[1].plot_tree()
