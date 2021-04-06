from decisiontree.tree import DecisionTree, plot_tree
from decisiontree.utils import column, remove_column

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
