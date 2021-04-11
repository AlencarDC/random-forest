import random

def get_bootstrap(x,y, seed=42):
  random.seed(seed)
  len_x = len(x)

  bootstrap_x = []
  bootstrap_y = []

  for i in range(len_x):
    sample_i = random.randint(0, len_x - 1)
    bootstrap_x.append(x[sample_i])
    bootstrap_y.append(y[sample_i])

  return (bootstrap_x, bootstrap_y)
