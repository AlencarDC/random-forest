from bootstrap import get_bootstrap
from decisiontree import DecisionTree, FeatureType
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

features_and_types = {
  "Tempo": FeatureType.CATEGORICAL,
  "Temperatura": FeatureType.CATEGORICAL,
  "Umidade": FeatureType.CATEGORICAL,
  "Ventoso": FeatureType.CATEGORICAL
}

def train_forest(x, y, num_trees, features_and_types):
    trees = []
    m_attributes = int(math.sqrt(len(x[0])))
    print("M attributes: ", m_attributes)
    for i in range(num_trees):
        tree = DecisionTree()
        training_set = get_bootstrap(x,y)
        tree.build(training_set[0], training_set[1], features_and_types, m_attributes)
        trees.append(tree)

    return trees

trees = train_forest(x,y,2,features_and_types)
