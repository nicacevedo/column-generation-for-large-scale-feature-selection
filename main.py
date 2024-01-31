#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from os import cpu_count

from models import SCIKIT_LASSO, L_LASSO, CG_SOC1_upgrade, SOCP
# Lasso: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

# Read data
X = pd.read_csv('real-data-treated/usa_n3522_m2440_yr2010_filled.csv', index_col=0, header=0)
y = pd.read_csv('real-data/s&p_n3522_yr2010.csv', index_col=0, header=0)
companies = X.columns.to_numpy()
X.index = pd.to_datetime(X.index)
y.index = pd.to_datetime(y.index)
# only keep the same index of y and X (y is the target)
common_index = y.index.intersection(X.index)
X = X.loc[common_index].to_numpy()
y = y.loc[common_index].to_numpy().T[0]

# Subset of the data
X = X[:,:]
companies = companies[:]

# # Print correlations of X with target y, ordering 
# # by absolute value of correlation

# corrs = [['company', 'corr_with_s&p']]
# for i in range(X.shape[1]):
#     print(companies[i], np.corrcoef(X[:,i],y)[1,0])
#     corrs.append([companies[i], np.corrcoef(X[:,i],y)[1,0]])

# corrs = pd.DataFrame(corrs[1:], columns=corrs[0])
# corrs = corrs.sort_values(by='corr_with_s&p', ascending=False)
# print(corrs[corrs['corr_with_s&p'].abs()>0.8999].to_latex(index=False))

# # plot y vs APH	(i=836)
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:,836], y, s=1)
# plt.xlabel('APH')
# plt.ylabel('s&p')
# plt.title('y vs APH')
# plt.show()

# # Plot both in time, standardizing both
# plt.figure(figsize=(12, 8))
# plt.plot((X[:,836] - np.mean(X[:,836]))/np.std(X[:,836]), label='APH')
# plt.plot((y - np.mean(y))/np.std(y), label='s&p')
# plt.legend()
# plt.show()

# # same for NCMI (i=1684)
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:,1684], y, s=1)
# plt.xlabel('NCMI')
# plt.ylabel('s&p')
# plt.title('y vs NCMI')
# plt.show()

# # Plot both in time, standardizing both
# plt.figure(figsize=(12, 8))
# plt.plot((X[:,1684] - np.mean(X[:,1684]))/np.std(X[:,1684]), label='NCMI')
# plt.plot((y - np.mean(y))/np.std(y), label='s&p')
# plt.legend()
# plt.show()




# print(X.corrwith(y).abs().sort_values(ascending=False))
# print(y.corr().abs().sort_values(by='s&p', ascending=False))
# display(X.loc[common_index].corr())

# Limit number of decimals to avoid numerical issues
# n_decimals = 6
# X = np.round(X, n_decimals)
# y = np.round(y, n_decimals)

# OLS regression
# add constant
# X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
# companies = np.concatenate((np.array(['constant']), companies))
beta_OLS = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_OLS

# Dimensions
n, m = X.shape


# Plot
plt.figure(figsize=(12, 12))
plt.scatter(y, y_hat, s=1)
plt.legend()
# plt.xlim([0, 5000])
# plt.ylim([0, 5000])
plt.show()

# Plot betas
thresh = 1e-0
plt.figure(figsize=(12, 6))
plt.bar(companies[np.abs(beta_OLS)>thresh], beta_OLS[np.abs(beta_OLS)>thresh])
# plt.xvalues = companies[beta_OLS>thresh]
plt.show()


# LASSO regression
init_tau = np.linalg.norm(y - y_hat) **2
n_selected_betas = []
model_errors = []
times = []
tol = 1e-6
n_samples = 10
for exp in np.linspace(2,7,50):
    tau = init_tau ** exp
    kappa = tau
    eps_sqrt = 0
    solver = cp.GUROBI
    # solver_params = {
    #     'mosek_params': {
    #     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':tol,
    #     'MSK_IPAR_NUM_THREADS': cpu_count(),
    #     },
    # }
    solver_params =  {
            'BarConvTol':tol,
            'BarQCPConvTol':tol
    }
    sample_times = []
    sample_errors = []
    sample_n_betas = []
    for i in range(n_samples):
        # beta_LASSO, z, lasso_value, time, p_time, r_dict = SCIKIT_LASSO(X, y, tau)
        beta, xi, u, z, lasso_value, time, p_time, r_dict = SOCP(X,y, tau, kappa, solver=solver, solver_params=solver_params, solver_verbose=True)
        # beta, xi, u, z, fo_value, time, p_time_mins, cg_dict = CG_SOC1_upgrade(
        #                         X, y, tau, kappa, eps_sqrt, cp.MOSEK, solver_params, solver_verbose=True, 
        #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
        #                         solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
        #                         v=int(m*0.012), v0=int(m*0.012), time_limit=60, dummy_condition=False,
        #                         canonic_random_initial_sol=True
        #                         )
        sample_times.append(time)
        y_hat = X @ beta
        sample_errors.append(np.linalg.norm(y - y_hat))
        sample_n_betas.append(np.sum(beta >= tol))

    n_selected_betas.append(np.mean(sample_n_betas))
    times.append(np.mean(sample_times))
    model_errors.append(np.mean(sample_errors))
    print(n_selected_betas)
    print(model_errors)
    print(times)

    # if np.linalg.norm(y - y_hat) **2 < 0.01:
    #     break

#%%

# Plot times in terms of number of selected betas
plt.figure(figsize=(12, 6))
plt.plot(n_selected_betas, times)
# add gray background to plot
plt.axvspan(0, 100, facecolor='gray', alpha=0.2)
plt.xlabel('Number of selected betas')
plt.ylabel('Time (s)')
plt.title('Time vs. number of selected betas (n, m = {}, {})'.format(n, m))
plt.show()

# Plot model errors in terms of number of selected betas
plt.figure(figsize=(12, 6))
plt.plot(n_selected_betas, model_errors)
plt.xlabel('Number of selected betas')
plt.ylabel('Model error (RMSE)')
plt.title('Model error vs. number of selected betas (n, m = {}, {})'.format(n, m))
plt.show()



# %%
