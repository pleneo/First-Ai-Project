import numpy as np
 # Ou 'Qt5Agg'
import matplotlib.pyplot as plt

# TAREFA DE REGRESSÃO.

# PONTO 1: CRIAR A VISUALIZAÇÃO BÁSICA DOS DADOS


"""
importa o .dat de aerogerador. 
Atualmente, esse data é um array de arrays, onde cada array, assim como no arquivo
.dat, possui 2 colunas e possui N linhas. As colunas são as variaveis x (independente,
ou seja, é o input) e a y (dependente, resultado do seu x correspondente)
Transforma o arquivo .dat em uma matriz que nesse caso é Nx2.
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
Having a good look at the generated plot, it is possible to observe the growing of the generated power as the wind speed
grows but at some point (~12) the growing of the generated power stops to increase and stagnates.
"""
plt.show()


# PONTO 2: ORGANIZAR DADOS

"""
reshape gives a new shape (dimension and structure) to some array without changing its data.
in this code, it formats a 1D vector to a column matrix Nx1 (2 dimensions). This will be used further
to be able to use matrix mathematical operations (multiply and transpose, i.e). (rewrite more detailed).
the -1 say to numpy find the total number of lines, the 1 means one column.

The reshape is needed because initially, this is a vector (unidimensional) and the reshape transforms it in a matrix 
R^Nx1.
"""
y_matrix = y.reshape(-1, 1)
x_matrix = x.reshape(-1, 1)

"""
Creates a matrix of ones using x_matrix number of lines and 1 column just like x_matrix

It will be used to concatenate the ones matrix with the x_matrix matrix and find the intercept (bias) of the model

The intercept is the "start" point of the model. It is a line that determine where your data will start when x = 0.

In a linear function, the equation is y = ax + b, with a being the angular coefficient and b being the linear coefficient
or *intercept*. THe angular coefficient determine the gradient (slope) of the line, and the intercept determine the 
starting point of these line when x = 0. If b does not exist, x will always have his start point in 0 in y axis.

Without the intercept, the regression line preticted by the model will necessarily start in 0. It is a problem because
if the model values does not really start in 0, the regression line will try to adjust to this values, but because the
line starts in 0, it will have a way bigger gradient, creating a way bigger MSE (Mean Squared Error) because the prediction
line does not consider properly the "center" of the dataset.

With the intercept, the model will be able to focus in find the right slope to the independent variables given the data
trend. The intercept will only care with the height of the line.

Because of the intercept, the model will have a R^p+1x1 (p+1 with p being the number of independent variables and + 1 is the
beta_0 who represents the intercept)

"""
ones = np.ones((x_matrix.shape[0], 1))

"""
In this step, the ones matrix is concatenated with the x matrix using hstack (horizontal stack), 
creating a Nx2 matrix.
Again, this will be used to find the intercept of the model.
"""
x_with_ones = np.hstack((ones, x_matrix))



# PONTO 3: IMPLEMENTANDO OS MODELOS MQO, MQO TIKHONOV E MÉDIA DA VARIÁVEL DEPENDENTE.

# 3.1: MQO Tradicional


"""
The operation (x.T @ x) condenses the data into an invertible square matrix.
The inversion of this matrix makes it possible to isolate the beta vector, while the (@ x.T @ y) term 
projects the observed power onto the feature space to find the slope and intercept that minimize 
the sum of squared residuals.

error: difference between data and reality
residual: difference between data and predicted model.

the tradicional LQS fully trust in the data, causing big imprecisions in the estimated line when the data noise is high.
"""

def treino_mqo_tradicional(x, y):
    beta_hat = np.linalg.inv(x.T @ x)@x.T@y
    return beta_hat

beta_hat = treino_mqo_tradicional(x_with_ones, y_matrix)
print('')



# 3.2 MQO Regularizado (Tikhonov)
"""

Adds lamb * I to the matrix introduces a penality to huge weights, forcing them to be smaller.
(do more research about it)

"""
def treino_mqo_regularizado(x,y, lamb):
    I = np.eye(x.shape[1])

    beta_hat = np.linalg.inv(x.T@x + lamb*I) @ x.T @ y

    return beta_hat


train_lambdas = [0, 0.25, 0.5, 0.75, 1]
models = {}

for i in train_lambdas:
    models[i] = treino_mqo_regularizado(x_with_ones, y_matrix, i)
