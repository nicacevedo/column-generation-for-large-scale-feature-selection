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


per_error = 0.1 # % error

# Size of the data
m_real = [(2*i+1)*1000 for i in range(0,3)]
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
        betas = np.random.uniform(-1,1, n_var)
        # # Random sign of each beta
        # betas = betas * np.random.choice([-1,1], n_var)
        y = pd.DataFrame(np.round(data @ betas, 3))

        # Change n_noise_cols random columns of X to a N(0,1) distribution
        noise_columns = np.random.choice(X.columns, n_noise_cols[i], replace=False)
        X[noise_columns] = np.random.normal(0, 1, (n_obs, n_noise_cols[i]))

        # OLS Solution
        beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)
        print(np.linalg.norm(beta_ols))


        # Prediction
        y_hat = X @ beta_ols

        # Plot real vs predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y, y_hat)
        plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        plt.show()

        # Original
        np.savetxt(f'./synthetic-data-error/synthetic_X_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', X, delimiter=',', fmt='%.3f')
        np.savetxt(f'./synthetic-data-error/synthetic_y_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', y, delimiter=',', fmt='%.3f')

        # Save the noise columns for the new data
        np.savetxt(f'./synthetic-data-error/Noise/synthetic_noise_columns_{m_real[i]}_{n_obs}_e{int(100*per_error)}.csv', np.sort(noise_columns), delimiter=',', fmt='%d')


def pairwise_correlated_data_generation(m_real, n_cols, n_obs_, corr_coef, random_state=123):
    n_noise_cols = [m_real[i] - n_cols[i] for i in range(len(n_cols))] 

    # Multiple instances with different number of columns
    for i,n_var in enumerate(m_real):
        n_partitions = n_var // 2

        n_obs = n_obs_[0] # CAUTION
        # for j, n_obs in enumerate(n_obs_):
        print(f'Generating data for {n_var} variables and {n_obs} observations')

        # List of column indexes
        columns = list(range(n_var))  
        # Partition the columns in random groups of n_partitions
        np.random.shuffle(columns)
        partitions = np.array_split(columns, n_partitions)

        # Data generation
        # np.random.seed(seed=random_state)
        data = np.zeros((n_obs, n_var))
        # print("corre coef", corr_coef)
        for partition in partitions:
            # Different mean and std for each partition
            mean = np.random.uniform(-10,10)
            std  = np.random.uniform(0, 5)
            j_0,j_1 = partition
            if len(partition) == 1:
                # Border case
                data[:,j_0]  = np.random.normal(mean, std, n_obs)
            else:
                # First random variable
                data[:,j_0] = np.random.normal(mean, std, n_obs)
                # Second random variable
                random_sign = np.random.choice([-1,1])
                data[:,j_1]  = (random_sign * corr_coef) * data[:,j_0] + np.sqrt(1-corr_coef**2) * np.random.normal(mean, std, n_obs) 
                # print("corr coef", np.corrcoef(data[:,j_0], data[:,j_1])[0,1])
                # Note: We can also set a random +/- sign for the second term, but it should not matter because each value of the 
                #       random normal distribution may have a different sign.

        # Model 
        X = pd.DataFrame(np.round(data,3)) # [:,:-n_noise_cols[i]]
        # print(np.round(np.corrcoef(X.T),2))
        # print(np.corrcoef(X.T).shape)
        # Random sign of each beta
        betas = np.random.uniform(-1,1, n_var)
        y = pd.DataFrame(np.round(data @ betas, 3))

        # Change n_noise_cols random columns of X to a N(0,1) distribution
        noise_columns = np.random.choice(X.columns, n_noise_cols[i], replace=False)
        X[noise_columns] = np.random.normal(0, 1, (n_obs, n_noise_cols[i]))

        # OLS Solution
        beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)
        print(np.linalg.norm(beta_ols))


        # Prediction
        y_hat = X @ beta_ols

        # # Plot real vs predicted
        # plt.figure(figsize=(6,6))
        # plt.scatter(y, y_hat)
        # plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        # plt.show()

        # Same plot, but a sample of points
        sample = np.random.choice(n_obs, 500)
        plt.figure(figsize=(6,6))
        plt.scatter(y.iloc[sample], y_hat.iloc[sample])
        plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        plt.show()

        # Original
        np.savetxt(f'./synthetic-data-correlated/synthetic_X_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', X, delimiter=',', fmt='%.3f')
        np.savetxt(f'./synthetic-data-correlated/synthetic_y_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', y, delimiter=',', fmt='%.3f')

        # Save the noise columns for the new data
        np.savetxt(f'./synthetic-data-correlated/Noise/synthetic_noise_columns_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', np.sort(noise_columns), delimiter=',', fmt='%d')


def correlated_data_generation(m_real, n_cols, n_obs_, corr_coef, random_state=42):
    n_noise_cols = [m_real[i] - n_cols[i] for i in range(len(n_cols))] 
    # print(n_noise_cols)
    # Multiple instances with different number of columns
    for i,n_var in enumerate(m_real):

        n_obs = n_obs_[0] # CAUTION
        # for j, n_obs in enumerate(n_obs_):
        print(f'Generating data for {n_var} variables and {n_obs} observations')

        # Data generation
        np.random.seed(seed=n_var) # Same seed for all m sizes
        # 1. Variance generation
        stds = np.random.uniform(0, 5,  size=n_var)
        variances = stds**2
        cov_matrix = np.diag(variances)
        # 2. Add covariance as cov(i,j) = std(i)*std(j)*corr_coef
        for k in range(n_var):
            for j in range(n_var):
                if k < j: # Only the upper triangle
                    # rand_sign = np.random.choice([-1,1]) # so the correlation can be negative also
                    cov_matrix[k,j] = stds[k]*stds[j]*corr_coef # *rand_sign
                    cov_matrix[j,k] = cov_matrix[k,j] # Symmetric matrix
        # Note: random sign affects the correlation matrix when the magnitude of the correlation is high. 
        # It's a symmetric matrix, but is not the theoretical correlation matrix that we want.
        # My explanation is that we cannot have: 
        #   cov(i,j) =  cov(j,k)
        #   cov(i,j) = -cov(i,k)
        #   cov(j,k) =  cov(i,k) [contradiction]
        # It can be solved selecting the correct sign for partitions.


        print("-"*50)
        # print("Covariance matrix")
        # print(cov_matrix)
        # theo_corr = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
        # print("Theoretical correlation matrix")
        # print(theo_corr)
        # 3. Generate multivariate normal matrix with the covariance matrix
        np.random.seed(seed=n_var) # Same seed for all m sizes
        data = np.random.multivariate_normal(np.zeros(n_var), cov_matrix, n_obs) # mu = 0, but it does not matter because of the later standardization
        # Print correlation matrix of data
        print("Correlation matrix")
        print(np.corrcoef(data.T))

    
        # Model (old y, without weights of betas)
        X = pd.DataFrame(np.round(data,3)) # [:,:-n_noise_cols[i]]
        np.random.seed(seed=n_var) # Same seed for all m sizes
        betas = np.random.uniform(-1,1, n_var)
        # # Random sign of each beta
        # betas = betas * np.random.choice([-1,1], n_var)
        y = pd.DataFrame(np.round(data @ betas, 3))

        # Change n_noise_cols random columns of X to a N(0,1) distribution
        np.random.seed(seed=n_var) # Same seed for all m sizes
        noise_columns = np.random.choice(X.columns, n_noise_cols[i], replace=False)
        X[noise_columns] = np.random.normal(0, 1, (n_obs, n_noise_cols[i]))

        # OLS Solution
        beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)
        print(np.linalg.norm(beta_ols))


        # Prediction
        y_hat = X @ beta_ols

        # Sample of indices
        sample = np.random.choice(n_obs, 500)

        # Plot real vs predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y.iloc[sample], y_hat.iloc[sample])
        plt.title(f'Original: {m_real[i]} variables, {n_obs} observations')
        plt.show()

        # Original
        np.savetxt(f'./synthetic-data-correlated/synthetic_X_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', X, delimiter=',', fmt='%.3f')
        np.savetxt(f'./synthetic-data-correlated/synthetic_y_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', y, delimiter=',', fmt='%.3f')

        # Save the noise columns for the new data
        np.savetxt(f'./synthetic-data-correlated/Noise/synthetic_noise_columns_{m_real[i]}_{n_obs}_e{int(100*per_error)}_corr{corr_coef}.csv', np.sort(noise_columns), delimiter=',', fmt='%d')

# # Normal data
# original_data_generation(m_real, n_cols, n_obs_, random_state=None)

# Correlated data
for corr_coef in [0.1, 0.3, 0.5, 0.7, 0.9]: 
    correlated_data_generation(m_real, n_cols, n_obs_, corr_coef, random_state=None)
# pairwise_correlated_data_generation(m_real, n_cols, n_obs_, corr_coef, random_state=None)

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
