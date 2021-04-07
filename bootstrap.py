import random
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

def get_bootstrap(x,y):
    len_x = len(x)

    bootstrap_x = []
    bootstrap_y = []
    
    for i in range(len_x):
        sample_i = random.randint(0, len_x - 1)
        bootstrap_x.append(x[sample_i])
        bootstrap_y.append(y[sample_i])

    return (np.array(bootstrap_x), np.array(bootstrap_y))

if __name__ == "__main__":
  x = data[:,0:4]
  y = data[:, 4]
  bootstrap = get_bootstrap(x,y)
  print(bootstrap[0])
  print(bootstrap[1])


