from kfold import kfold

features = ["Index", "Tempo","Temperatura","Umidade","Ventoso"]
data = [
  [1,"Ensolarado","Quente","Alta","Falso","Nao"],
  [2, "Ensolarado","Quente","Alta","Verdadeiro","Nao"],
  [3,"Nublado","Quente","Alta","Falso","Sim"],
  [4,"Chuvoso","Amena","Alta","Falso","Sim"],
  [5,"Chuvoso","Fria","Normal","Falso","Sim"],
  [6,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [7,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [8,"Ensolarado","Amena","Alta","Falso","Nao"],
  [9,"Ensolarado","Fria","Normal","Falso","Sim"],
  [10,"Chuvoso","Amena","Normal","Falso","Sim"],
  [11,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [12,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [13,"Nublado","Quente","Normal","Falso","Sim"],
  [14,"Chuvoso","Fria","Normal","Falso","Sim"],
  [15,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [16,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [17,"Ensolarado","Amena","Alta","Falso","Nao"],
  [18,"Ensolarado","Fria","Normal","Falso","Sim"],
  [19,"Chuvoso","Amena","Normal","Falso","Sim"],
  [20,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [21,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [22,"Nublado","Quente","Normal","Falso","Sim"],
  [23,"Chuvoso","Fria","Normal","Falso","Sim"],
  [24,"Chuvoso","Fria","Normal","Verdadeiro","Nao"],
  [25,"Nublado","Fria","Normal","Verdadeiro","Sim"],
  [26,"Ensolarado","Amena","Alta","Falso","Nao"],
  [27,"Ensolarado","Fria","Normal","Falso","Sim"],
  [28,"Chuvoso","Amena","Normal","Falso","Sim"],
  [29,"Ensolarado","Amena","Normal","Verdadeiro","Sim"],
  [30,"Nublado","Amena","Alta","Verdadeiro","Sim"],
  [31,"Nublado","Quente","Normal","Falso","Sim"],
  [32,"Chuvoso","Amena","Alta","Verdadeiro","Nao"]
]


print(kfold(10, data, features))
