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

fig = plt.figure()
ax = fig.add_subplot()

# Define as propriedades básicas de título do subplot ax
ax.set_xlabel("Velocidade do Vento",  fontsize=12)
ax.set_ylabel("Potência Gerada",  fontsize=12)
plt.title('Distribuição dos Dados do Aerogerador', fontsize=14)
ax.scatter(x,y, c="cyan", edgecolor='k')


plt.show()
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
def imprimir_tabela_final(r_w, r_wo):
    modelos = [
        ("Média da variável dependente", r_w, 0),
        ("MQO tradicional (Com Intercepto)", r_w, 1),
        ("MQO sem intercepto (Físico)", r_wo, 1),  # Sua análise extra
        ("MQO regularizado (0,25)", r_w, 2),
        ("MQO regularizado (0,50)", r_w, 3),
        ("MQO regularizado (0,75)", r_w, 4),
        ("MQO regularizado (1,00)", r_w, 5)
    ]

    print(f"\n{'MODELOS':<35} | {'MÉDIA R²':<10} | {'DESVIO':<10} | {'MAIOR':<10} | {'MENOR':<10}")
    print("-" * 85)
    for nome, fonte, idx in modelos:
        print(
            f"{nome:<35} | {fonte['m_r2'][idx]:<10.4f} | {fonte['s_r2'][idx]:<10.4f} | {fonte['max_r2'][idx]:<10.4f} | {fonte['min_r2'][idx]:<10.4f}")
    print("-" * 85)


imprimir_tabela_final(res_w, res_wo)

# --- PONTO 6: VISUALIZAÇÃO CONSOLIDADA ---
plt.figure(figsize=(12, 7))

# 1. Dados Reais
plt.scatter(x, y, c="cyan", alpha=0.4, edgecolor='k', s=20, label="Dados Experimentais")

# Gerando pontos para as retas
x_plot = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_plot_w = np.hstack((np.ones((100, 1)), x_plot))

# 2. Linha da Média (Baseline obrigatório) [cite: 17]
plt.axhline(y=np.mean(y), color='gray', linestyle='--', linewidth=2, label="Média da Var. Dep.")

# 3. MQO Tradicional (λ=0 com Intercepto) [cite: 18, 21]
plt.plot(x_plot, x_plot_w @ models_with_ones[0], 'r-', linewidth=2.5, label="MQO Tradicional")

# 4. Seu toque de mestre: Reta sem intercepto (Física)
plt.plot(x_plot, x_plot @ models_without_ones[0], 'k:', linewidth=2.5, label="MQO s/ Intercepto")

# 5. Tikhonov (Exemplo com λ=1.0) [cite: 20, 21]
plt.plot(x_plot, x_plot_w @ models_with_ones[1], 'purple', linestyle='--', linewidth=2, label="Tikhonov (λ=1.0)")

plt.title('Comparativo de Modelos de Treinamento do Aerogerador', fontsize=14)
plt.xlabel('Velocidade do Vento', fontsize=12)
plt.ylabel('Potência Gerada', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()