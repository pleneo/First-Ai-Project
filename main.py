import numpy as np
 # Ou 'Qt5Agg'
import matplotlib.pyplot as plt

# TAREFA DE REGRESSÃO.

# PONTO 1: CRIAR A VISUALIZAÇÃO BÁSICA DOS DADOS


"""
importa o .dat de aerogerador. 
Atualmente, esse data é um array de arrays, onde cada array, assim como no arquivo
.dat, possui 2 colunas e possui N linhas. As colunas são as variaveis x (independente,
ou seja, é o input) e a y (dependente, resultado do seu x correspondente
"""
data = np.loadtxt("aerogerador.dat")

"""
o ':' representa que todas as linhas serão injetadas na variável criada.

o zero após a vírgula é um índice significa que será considerado apenas a coluna 0
do aerogerador.dat.
"""
x = data[:, 0]
y = data[:, 1]

"""
Nas linhas abaixo, é criado o objeto figure, que é uma janela que será aberta.

o ax criado a partir do add_subplot é criar um gráfico dentro de figure, onde a partir
dele que será criado toda a estrutura do gráfico a ser plotado.

Uma figure pode ter vários subplots.
"""
fig = plt.figure()
ax = fig.add_subplot()

# Define as propriedades básicas de título do subplot ax
ax.set_xlabel("Velocidade")
ax.set_ylabel("Potência")

"""
O scatter é um modelo de visualização de dados que é voltado a observação da correlação entre
variáveis.
No caso desse trabalho, o scatter está sendo utilizado para observar qual a relação entre x e y,
ou seja, o que acontece com y a partir de dados x.
In the context of supervised learning, it is essencial to use the scatter to observe these correlations.
"""
ax.scatter(x,y, c="cyan", edgecolor='k')

"""
In this plot, was observed a similarity between the curve made in the visualization of this plot
and a logarithmic curve.
How can I explain it?
"""
plt.show()


# PONTO 2: ORGANIZAR DADOS

"""
reshape gives a new shape (dimension and structure) to some array without changing its data.
in this code, it formats a 1 line N columns array/vector to a matrix Nx1. This will be used further
to be able to use matrix mathematical operations. (rewrite more detailed).
the -1 say to numpy find the total number of lines, the 1 means one column.
"""
y_matrix = y.reshape(-1, 1)
x_matrix = x.reshape(-1, 1)

"""
Creates a matrix of ones using x_matrix number of lines and 1 column just like x_matrix

It will be used to concatenate the ones matrix with the x_matrix matrix and find the intercept (bias) of the model
(i think i need to made more research about intercept and these ones matrix to have a more deep
knowledge)
"""
ones = np.ones((x_matrix.shape[0], 1))

"""
In this step, the ones matrix is concatenated with the x matrix using hstack (horizontal stack), 
creating a Nx2 matrix.
Again, this will be used to find the intercept of the model.
"""
x_with_ones = np.hstack((ones, x_matrix))



# PONTO 3: IMPLEMENTANDO OS MODELOS MQO, MQO TIKHONOV E MÉDIA DA VARIÁVEL DEPENDENTE.


