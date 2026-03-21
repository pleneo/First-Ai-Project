import numpy as np
import matplotlib.pyplot as plt

# --- PONTO 1 & 2: CARREGAMENTO E ORGANIZAÇÃO ---
data = np.loadtxt("aerogerador.dat")
x = data[:, 0]
y = data[:, 1]

y_matrix = y.reshape(-1, 1)
x_matrix = x.reshape(-1, 1)

ones = np.ones((x_matrix.shape[0], 1))
x_with_ones = np.hstack((ones, x_matrix))
x_without_ones = x_matrix  # Apenas a velocidade


# --- PONTO 3: FUNÇÕES DE TREINO ---

def treino_mqo_regularizado(x, y, lamb):
    I = np.eye(x.shape[1])
    # Só zeramos o primeiro índice da identidade se houver intercepto (2 colunas)
    if x.shape[1] > 1:
        I[0, 0] = 0

    beta_hat = np.linalg.inv(x.T @ x + lamb * I) @ x.T @ y
    return beta_hat


# --- PONTO 4: TREINO DOS MODELOS (Dataset Completo) ---
train_lambdas = [0, 0.25, 0.5, 0.75, 1]
models_with_ones = {}
models_without_ones = {}

for i in train_lambdas:
    models_with_ones[i] = treino_mqo_regularizado(x_with_ones, y_matrix, i)
    models_without_ones[i] = treino_mqo_regularizado(x_without_ones, y_matrix, i)

# --- PONTO 5: RANDOM SUBSAMPLING VALIDATION ---
percentual_de_treino = .8
R = 500

# Inicialização das matrizes de resultados
resultados_mse_with_ones = np.zeros((R, 2 + len(train_lambdas[1:])))
resultados_r2_with_ones = np.zeros((R, 2 + len(train_lambdas[1:])))

resultados_mse_without_ones = np.zeros((R, 2 + len(train_lambdas[1:])))
resultados_r2_without_ones = np.zeros((R, 2 + len(train_lambdas[1:])))

for i in range(R):
    indices = np.random.permutation(len(y_matrix))
    corte = int(len(y_matrix) * percentual_de_treino)
    idx_tr, idx_te = indices[:corte], indices[corte:]

    # Dados de treino e teste para ambas as versões
    y_tr, y_te = y_matrix[idx_tr], y_matrix[idx_te]

    x_tr_w, x_te_w = x_with_ones[idx_tr], x_with_ones[idx_te]
    x_tr_wo, x_te_wo = x_without_ones[idx_tr], x_without_ones[idx_te]

    sst = np.sum((y_te - np.mean(y_te)) ** 2)
    y_pred_media = np.mean(y_tr)
    ssr_media = np.sum((y_te - y_pred_media) ** 2)

    # Baseline (Média) é igual para ambos
    for mat_mse, mat_r2 in [(resultados_mse_with_ones, resultados_r2_with_ones),
                            (resultados_mse_without_ones, resultados_r2_without_ones)]:
        mat_mse[i, 0] = ssr_media / len(y_te)
        mat_r2[i, 0] = 1 - (ssr_media / sst)

    # Loop para cada Lambda
    for j, lamb in enumerate(train_lambdas):
        # 1. Com Intercepto
        beta_w = treino_mqo_regularizado(x_tr_w, y_tr, lamb)
        y_pred_w = x_te_w @ beta_w
        ssr_w = np.sum((y_te - y_pred_w) ** 2)
        resultados_mse_with_ones[i, j + 1] = ssr_w / len(y_te)
        resultados_r2_with_ones[i, j + 1] = 1 - (ssr_w / sst)

        # 2. Sem Intercepto
        beta_wo = treino_mqo_regularizado(x_tr_wo, y_tr, lamb)
        y_pred_wo = x_te_wo @ beta_wo
        ssr_wo = np.sum((y_te - y_pred_wo) ** 2)
        resultados_mse_without_ones[i, j + 1] = ssr_wo / len(y_te)
        resultados_r2_without_ones[i, j + 1] = 1 - (ssr_wo / sst)


# --- PROCESSAMENTO ESTATÍSTICO ---
def calcular_metricas(mse_mat, r2_mat):
    return {
        'm_mse': np.mean(mse_mat, axis=0), 's_mse': np.std(mse_mat, axis=0),
        'min_mse': np.min(mse_mat, axis=0), 'max_mse': np.max(mse_mat, axis=0),
        'm_r2': np.mean(r2_mat, axis=0), 's_r2': np.std(r2_mat, axis=0),
        'min_r2': np.min(r2_mat, axis=0), 'max_r2': np.max(r2_mat, axis=0)
    }


res_w = calcular_metricas(resultados_mse_with_ones, resultados_r2_with_ones)
res_wo = calcular_metricas(resultados_mse_without_ones, resultados_r2_without_ones)

# --- APRESENTAÇÃO DOS RESULTADOS ---
labels = ["Média", "MQO (L=0)", "L=0.25", "L=0.5", "L=0.75", "L=1.0"]


def imprimir_tabela(titulo, r):
    print(f"\n{'=' * 20} {titulo} {'=' * 20}")
    print(f"{'MODELO':<12} | {'MÉTRICA':<8} | {'MÉDIA':<10} | {'DESVIO':<10} | {'MENOR':<10} | {'MAIOR':<10}")
    print("-" * 95)
    for i, lab in enumerate(labels):
        print(
            f"{lab:<12} | {'MSE':<8} | {r['m_mse'][i]:<10.4f} | {r['s_mse'][i]:<10.4f} | {r['min_mse'][i]:<10.4f} | {r['max_mse'][i]:<10.4f}")
        print(
            f"{'':<12} | {'R²':<8} | {r['m_r2'][i]:<10.4f} | {r['s_r2'][i]:<10.4f} | {r['min_r2'][i]:<10.4f} | {r['max_r2'][i]:<10.4f}")
        print("-" * 95)


imprimir_tabela("COM INTERCEPTO (Função Afim)", res_w)
imprimir_tabela("SEM INTERCEPTO (Regressão na Origem)", res_wo)

# --- PONTO 6: VISUALIZAÇÃO ---

# Figura 1: Com Intercepto
plt.figure(figsize=(10, 5))
plt.scatter(x, y, c="cyan", alpha=0.3, label="Dados", edgecolor='k')
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_w = np.hstack((np.ones((100, 1)), x_range))
plt.plot(x_range, x_range_w @ models_with_ones[0], 'r-', linewidth=3, label="MQO (Com Intercepto)")
plt.title("Modelo com Intercepto")
plt.legend();
plt.grid(True, alpha=0.3)

# Figura 2: Sem Intercepto
plt.figure(figsize=(10, 5))
plt.scatter(x, y, c="cyan", alpha=0.3, label="Dados", edgecolor='k')
plt.plot(x_range, x_range @ models_without_ones[0], 'g-', linewidth=3, label="MQO (Sem Intercepto)")
plt.title("Modelo sem Intercepto 1")
plt.legend();
plt.grid(True, alpha=0.3)

plt.show()