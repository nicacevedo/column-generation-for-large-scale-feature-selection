#%%
# TO DO (29/06/2023): 
# 1. Lo primero es que agregué una ponderación a cada dist. normal
# de las columnas de X. Antes estaba sólo beta=1 para todos.
# 2. Lo segundo es que hay dos maneras que se me ocurren en este momento de crear
# una buena base de datos sintética: 
# 2.1) Tomar y = X_real @ beta. Mientras que X = sub(X_real) U ruido (ruido con dist. por definir) 
# 2.2) Tomar y = X_real @ beta. Mientras que X = X_real U ruido, y la idea
# es que el ruido no se tome. [Caso particular de 2.1 con per_error = 0].

# Me arrepiento de lo que dije arriba. Debería tener dos métricas a elegir.
# 1. El % de columnas originales que borro.
# 2. El % de columnas que agrego con ruido.
# Luego, crear una base de datos sintética con cada combinación de métricas.

# New File (05/08/2023): Todo mal. Debe ser el mismo % que borro y agrego ruido. Así mantengo dimensiones originales 
# también se puede ver claro cuáles deben ser eliminadas.

# Suma de normales
import numpy as np
import matplotlib.pyplot as plt

import pickle
import pandas as pd


per_error = 0.2 # % error

# Size of the data
m_real = [(i+1)*1000 for i in range(0,10)]
n_cols = [int(np.round(x*(1-per_error))) for x in m_real]
n_obs_ = [10000]

# New version
def original_data_generation(m_real, n_cols, n_obs_, random_state=123):
    n_noise_cols = [m_real[i] - n_cols[i] for i in range(len(n_cols))] 
    # print(n_noise_cols)
    # Multiple instances with different number of columns
    for i,n_var in enumerate(m_real):

        n_obs = n_obs_[0] # CAUTION
        # for j, n_obs in enumerate(n_obs_):
        print(f'Generating data for {n_var} variables and {n_obs} observations')

        # Data generation
        # np.random.seed(seed=random_state)
        data = np.zeros((n_obs, n_var))
        for j in range(n_var):
            mean = np.random.uniform(-10,10)
            std  = np.random.uniform(0, 5)
            data[:,j]  = np.random.normal(mean, std, n_obs)

        # Model (old y, without weights of betas)
        X = pd.DataFrame(np.round(data,3)) # [:,:-n_noise_cols[i]]
        betas = np.random.uniform(-10,10, n_var)
        y = pd.DataFrame(np.round(data @ betas, 3))
        # y = pd.DataFrame(np.round(np.sum(data, axis=1),3))

        # Change n_noise_cols random columns of X to a N(0,1) distribution
        noise_columns = np.random.choice(X.columns, n_noise_cols[i], replace=False)
        # print(noise_columns)
        # print(X.shape)
        X[noise_columns] = np.random.normal(0, 1, (n_obs, n_noise_cols[i]))
        # print(X.shape)
        # # New synthetic: y = X_data @ betas
        # betas = np.random.uniform(-10,10, n_var)
        # y = pd.DataFrame(np.round(data @ betas, 3))
        # plt.hist(y, bins=50)
        # plt.show()

        # # OLS Solution
        # beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)
        # X2 = X.iloc[:,:-n_noise_cols[i]]
        # # print(X2.shape)
        # beta_ols2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ y) 

        # # Prediction
        # y_hat = X @ beta_ols
        # y_hat2 = X2 @ beta_ols2
        # # print(y)
        # # print(y_hat)

        # # Plot real vs predicted
        # plt.figure(figsize=(6,6))
        # plt.scatter(y, y_hat)
        # plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        # plt.show()

        # # Plot real vs predicted
        # plt.figure(figsize=(6,6))
        # plt.scatter(y, y_hat2, color='red')
        # plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        # plt.show()

        # Original
        np.savetxt(f'./synthetic-data-error/synthetic_X_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', X, delimiter=',', fmt='%.3f')
        np.savetxt(f'./synthetic-data-error/synthetic_y_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', y, delimiter=',', fmt='%.3f')

        # Save the noise columns for the new data
        np.savetxt(f'./synthetic-data-error/Noise/synthetic_noise_columns_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', np.sort(noise_columns), delimiter=',', fmt='%d')


original_data_generation(m_real, n_cols, n_obs_, random_state=0)


# def new_data_generation(m_real, n_cols, n_obs, random_state=123):
#     n_var_errors = [m_real[i] - n_cols[i] for i in range(len(n_cols))] 

#     for i,n_var in enumerate(n_cols):

#         # Data
#         np.random.seed(seed=random_state)
#         data = np.zeros((n_obs, n_var))
#         for j in range(n_var):
#             mean = np.random.uniform(-10,10)
#             std  = np.random.uniform(0, 5)
#             data[:,j]  = np.random.normal(mean, std, n_obs)

#         # Model (old y, without weights of betas)
#         # # New synthetic: y = X_data @ betas
#         betas = np.random.uniform(-10,10, n_var)
#         y = pd.DataFrame(np.round(data @ betas, 3))
#         # y = pd.DataFrame(np.round(np.sum(data, axis=1),3))
#         print((n_obs, n_var_errors[i]))
#         print(data.shape)
#         data[:,:-n_var_errors[i]] = np.random.normal(0, 1, (n_obs, n_var_errors[i]))
#         X = pd.DataFrame(np.round(data,3))


#         plt.hist(y, bins=50)
#         plt.show()

#         # OLS Solution
#         beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

#         # Prediction
#         y_hat = X @ beta

#         # Plot real vs predicted
#         plt.figure(figsize=(6,6))
#         plt.scatter(y, y_hat)
#         plt.show()

#         # For analysis
#         np.savetxt(f'./Synthetic Data Analysis/Bases/syntetic_X_{m_real[i]}_{n_obs}.csv', X, delimiter=',', fmt='%.3f')
#         np.savetxt(f'./Synthetic Data Analysis/Bases/syntetic_y_{m_real[i]}_{n_obs}.csv', y, delimiter=',', fmt='%.3f')

# # print(m_real, n_cols, n_obs)
# new_data_generation(m_real, n_cols, n_obs, random_state=123)

# %%
