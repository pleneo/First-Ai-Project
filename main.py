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
# plt.show()


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

em portugues agora por que minha mente ta derretendo:

A função principal desse algortimo é encontrar uma reta que melhor se encaixe entre os pontos dados.
Com essa reta, é possível prever um valor de um novo x dado com uma margem de erro do modelo e do ruído dos dados
Nesse caso, o beta_hat na posição 0 B0 possui o intercepto e os outros valores acompanham os x (x1b1 + x2b2+...+xnbn)
O objetivo do MQO é minimizar a Soma dos Quadrados dos Resíduos  (somatorio dos (yi - ŷi)**2 com yi sendo o valor real
e ŷi sendo o valor previsto).

Nesse caso, o treinamento do mqo está sendo feito via equação normal. (o que é?)
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
    I[0, 0] = 0; # NAO REGULARIZAR O INTERCEPTO
    beta_hat = np.linalg.inv(x.T@x + lamb*I) @ x.T @ y

    return beta_hat

# 4. TESTE DO MODELO COM OS VALORES LAMBDA

train_lambdas = [0, 0.25, 0.5, 0.75, 1]
models = {}

for i in train_lambdas:
    models[i] = treino_mqo_regularizado(x_with_ones, y_matrix, i)

# 5. Random Subsampling Validation - Validação do Modelo


percentual_de_treino = .8
R = 500
resultados_mse = np.zeros((R, 2 + len(train_lambdas[1:])))
resultados_r2 = np.zeros((R, 2 + len(train_lambdas[1:])))

for i in range(R):
    indices = np.random.permutation(len(y_matrix))
    corte = int(len(y_matrix) * percentual_de_treino)

    idx_treino, idx_teste = indices[:corte], indices[corte:]

    x_treino, x_teste = x_with_ones[idx_treino], x_with_ones[idx_teste]
    y_treino, y_teste = y_matrix[idx_treino], y_matrix[idx_teste]

    sum_of_squares_total = np.sum((y_teste - np.mean(y_teste)) ** 2)

    y_predicao_media = np.mean(y_treino)
    ssr_media = np.sum((y_teste - y_predicao_media)**2)

    resultados_mse[i,0] = ssr_media / len(y_teste)
    resultados_r2[i, 0] = 1 - (ssr_media / sum_of_squares_total)

    for j, lamb in enumerate(train_lambdas):
        beta_hat1 = treino_mqo_regularizado(x_treino, y_treino, lamb)
        y_predicao = x_teste@beta_hat1

        ssr_modelo = np.sum((y_teste - y_predicao) ** 2)

        resultados_mse[i, j + 1] = ssr_modelo / len(y_teste)
        resultados_r2[i, j + 1] = 1 - (ssr_modelo / sum_of_squares_total)


media_mse  = np.mean(resultados_mse, axis=0)
desvio_mse = np.std(resultados_mse, axis=0)
maior_mse  = np.max(resultados_mse, axis=0)
menor_mse  = np.min(resultados_mse, axis=0)

# Para o R²
media_r2   = np.mean(resultados_r2, axis=0)
desvio_r2  = np.std(resultados_r2, axis=0)
maior_r2   = np.max(resultados_r2, axis=0)
menor_r2   = np.min(resultados_r2, axis=0)

# --- APRESENTAÇÃO DOS RESULTADOS (TABELA COMPLETA) ---

labels = ["Média", "MQO (L=0)", "L=0.25", "L=0.5", "L=0.75", "L=1.0"]

print("\n" + "="*95)
print(f"{'MODELO':<12} | {'MÉTRICA':<8} | {'MÉDIA':<10} | {'DESVIO':<10} | {'MENOR':<10} | {'MAIOR':<10}")
print("-" * 95)

for i, lab in enumerate(labels):
    # Linha do MSE
    print(f"{lab:<12} | {'MSE':<8} | {media_mse[i]:<10.4f} | {desvio_mse[i]:<10.4f} | {menor_mse[i]:<10.4f} | {maior_mse[i]:<10.4f}")
    # Linha do R²
    print(f"{'':<12} | {'R²':<8} | {media_r2[i]:<10.4f} | {desvio_r2[i]:<10.4f} | {menor_r2[i]:<10.4f} | {maior_r2[i]:<10.4f}")
    print("-" * 95)

# --- PONTO 6: VISUALIZAÇÃO FINAL (PLOT DAS RETAS) ---
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c="cyan", alpha=0.4, label="Dados Reais", edgecolor='k')

# Criar pontos para desenhar as retas
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_ones = np.hstack((np.ones((100, 1)), x_range))

# Reta do MQO Tradicional (L=0)
plt.plot(x_range, x_range_ones @ models[0], 'r-', linewidth=3, label="MQO Tradicional")

# Reta do Modelo de Média
plt.axhline(y=np.mean(y), color='black', linestyle='--', label="Modelo de Média")

plt.title("Comparação de Modelos")
plt.xlabel("Velocidade")
plt.ylabel("Potência")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
