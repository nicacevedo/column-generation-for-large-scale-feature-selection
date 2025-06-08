#%%
import cvxpy as cp
import numpy as np
import pandas as pd
import clarabel

# from time import time, process_time

import gurobipy as gp
from os import cpu_count

# from models import CG_SOC1_upgrade, SOCP, SCIKIT_ElasticNet, CG_SOC1_ElasticNet, CG_LASSO_SOC1, CG_LASSO_SOC1_v2, CG_LASSO_SOC2, SCIKIT_LASSO, L_LASSO, L_LASSO_scipy

#2025
from src.cg_models import CG_LASSO_SOC1_v2
# import mosek as MSK


if __name__ == "__main__":
    for corr_coef in [0.5]: # For synthetic correlated data: 0.1, 0.3, 0.5, 0.7, 
        # error %
        e = 10
        real_data = False

        #2025 
        unboundedness_policy = "unit_ball"

        out_file = f'results/unboundedness_policy/soc1_lasso_v2_{corr_coef}_{unboundedness_policy}.csv' # 2025
        # out_file = f'results/method-benchmark/main_methods_benchmark_corr{corr_coef}_data.csv'
        # Sample of the data
        final_results = [['model','n_row','m_col', 'solver', 'OLS_error_quad', 'error_quad', 'fo_value','time', 'time_process', 'k_iter', 'k_v_iter', 'n_betas','status', 'tau', 'kappa', 'lambda','alpha', 'theta', 'tau_exp']]
        # for n_row in [3522]: # s&p 500 
        for n_row in [10000]: 
        # for n_row in [10000]:#[91992]: # casen
            # for m_col in [500, 1000, 1500, 2000, 2440]: # s&p 500
            # for m_col in [100, 200, 300, 400, 445]:# casen
            for m_col in [1000]:
            # for m_col in [(2*i+1)*1000 for i in range(0,3)]:
                # 500, 1000, 1500, 2000, 2440

                if real_data:
                    # real data (S&P 500, USA data)
                    X = pd.read_csv('real-data-treated/usa_n3522_m2440_yr2010_filled.csv', index_col=0, header=0)
                    y = pd.read_csv('real-data/s&p_n3522_yr2010.csv', index_col=0, header=0)

                    # Datetime adjustment
                    companies = X.columns.to_numpy()
                    X.index = pd.to_datetime(X.index)
                    y.index = pd.to_datetime(y.index)
                    # only keep the same index of y and X (y is the target)
                    common_index = y.index.intersection(X.index)
                    X = X.loc[common_index].to_numpy()
                    y = y.loc[common_index].to_numpy().T[0]
        
                    # print(X.shape, y.shape)
                    X = X[:n_row,:m_col]
                    y = y[:n_row]

                else:
                    # # synthetic data, with the first row including the number of rows and columns
                    # X = pd.read_csv(f'synthetic-data-error/synthetic_X_{m_col}_{n_row}_e{e}.csv', delimiter=',', header=None).to_numpy()
                    # y = pd.read_csv(f'synthetic-data-error/synthetic_y_{m_col}_{n_row}_e{e}.csv', delimiter=',', header=None).to_numpy().T[0]

                    X = pd.read_csv(f'synthetic-data-correlated/synthetic_X_{m_col}_{n_row}_e{e}_corr{corr_coef}.csv', delimiter=',', header=None).to_numpy()
                    y = pd.read_csv(f'synthetic-data-correlated/synthetic_y_{m_col}_{n_row}_e{e}_corr{corr_coef}.csv', delimiter=',', header=None).to_numpy().T[0]

                    # # CAUTION: shuffle the columns of X, to avoid the original order (scikit-learn does sequential selection)
                    # np.random.shuffle(X.T) WASN'T BECAUSE OF THIS THA SCIKIT-LEARN WAS BETTER IN SYNTHETIC DATA
                

                # Dimension of the problem
                n,m = X.shape
                print("="*100)
                print(f'Number of rows: {n}, Number of columns: {m}, Correlation coef: {corr_coef}')
                # exit()

                # 


                # # Correlation matrix
                # corr = np.corrcoef(X.T)
                # print(corr)
                # exit()


                # print(np.where(np.std(X, axis=0) == 0))
                # print(np.where(np.isnan(X)))
                # print(np.any(np.isfinite(np.std(X, axis=0))))
                # print(np.any(np.isnan((X - np.mean(X, axis=0)))))
                # print(np.any(np.isfinite((X - np.mean(X, axis=0)))))                
                # exit()    
                # Standardize data
                X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
                y = (y - np.mean(y))/np.std(y)

                # # print(X)
                # # print(y)
                # print(np.where(np.isnan(X)))
                # # print(np.any(np.isfinite(X)))
                # exit()

                # # Condition number of the matrix X^T @ X
                # eigen_vals = np.linalg.eigvals(X.T @ X)
                # print("Condition number of the matrix X^T @ X:", eigen_vals.max()/eigen_vals.min())
                # # Condition number of the correlation matrix of X
                # corr = np.corrcoef(X.T)
                # eigen_vals = np.linalg.eigvals(corr)
                # print("Condition number of the correlation matrix of X:", eigen_vals.max()/eigen_vals.min())

                # OLS solution
                if n <= 40000:# Note: 16GB of RAM
                    beta_OLS = np.linalg.inv(X.T @ X) @ X.T @ y
                else:
                    # Approximation of the OLS solution
                    beta_OLS = np.linalg.pinv(X) @ y
                # print('OLS betas != 0:', beta_OLS[beta_OLS > 1e-6].size)
                # exit()
                error_quad_OLS = np.linalg.norm(y - X @ beta_OLS) **2

                # Params
                eps_sqrt = 0
                init_tau = error_quad_OLS


                # Skip that model in the loop if the last execution was more than models_time_limit
                models_time_limit = 15 # minutes
                last_execution_times = {
                    'SOCP': 0,
                    'CG_LASSO_SOC1_v2': 0,
                    'CG_LASSO_SOC1': 0,
                    'CG_LASSO_SOC2': 0,
                    'SCIKIT_LASSO': 0,
                    'L_LASSO': 0,
                    'L_LASSO_scipy': 0,
                    'CG_SOC1_upgrade': 0,
                    'CG_SOC1_ElasticNet': 0,
                    'CG_LASSO_SOC1_v2': 0,
                    'CG_LASSO_SOC1': 0,
                    'CG_LASSO_SOC2': 0,
                    'SCIKIT_LASSO': 0,
                    'L_LASSO': 0,
                    'L_LASSO_scipy': 0,
                }
                    

                # for tau_exp in np.linspace(-1.5, .1, 10): # all previuos experiments were with tau_exp = 7
                for tau_exp in np.linspace(-2, -1, 2): # -1/.15 for the other synthetic

                    print("="*100)
                    print(f'Number of rows: {n}, Number of columns: {m}, Correlation coef: {corr_coef}, Tau exp: {tau_exp}')

                # for tau_exp in np.linspace(1, 5, 10):

                # tau_exp_0, tau_exp_f = -1, 7
                # tau_tilda_0 = init_tau / m ** tau_exp_0
                # tau_tilda_f = init_tau / m ** tau_exp_f
                # for tau_tilda in np.linspace(tau_tilda_0, tau_tilda_f, 10):
                #     tau_exp = np.log(init_tau / tau_tilda ) / np.log(m) # inverse function to get the tau_exp
                #     for theta_exp in [-6,-3,0,3,6]:
                    theta_exp = 1
                    if True:
                        tau_tilda = init_tau/m ** tau_exp                      # 1/error bc it's a decimal number in most cases

                        # for alpha_ in [.2, .4, .6, .8,  1]:
                        alpha_ = 1
                        if True:
                            alpha = 1/alpha_
                            # tau = (1 - ratio) * tau_tilda
                            tau = tau_tilda / alpha 
                            # kappa =  
                            kappa = alpha ** 2 * tau
                            # print('tau2', tau, 'kappa', kappa, 'alpha', alpha)
                            # print('sqrt(tau*kappa)', np.sqrt(tau*kappa), 'sqrt(tau_tilda**2)', tau_tilda)
                            theta = init_tau ** theta_exp

                            solver_verbose = False
                            solvers = ['MOSEK']  #'GUROBI', 'MOSEK', 'COPT' 'ECOS', 'ECOS_BB', 'CLARABEL', 'OSQP', 'SCS', 'CVXOPT', 
                            add_constant = False
                            time_limit = 60#60*24*7 # 7 days
                            for solver in solvers:
                                for i in range(1):
                                    # solver = solvers[-1]
                                    tol = 1e-6
                                    solver_params_ =  {
                                        'GUROBI': {
                                            'BarConvTol':tol,
                                            'BarQCPConvTol':tol
                                        },
                                        'MOSEK': {
                                            'mosek_params':{
                                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':tol,
                                            'MSK_IPAR_NUM_THREADS': cpu_count(),
                                            }
                                        },
                                        'ECOS': {
                                            'abstol':tol,
                                            'reltol':tol,
                                            'feastol':tol,
                                        },
                                        'ECOS_BB': {
                                            'abstol':tol,
                                            'reltol':tol,
                                            'feastol':tol,
                                        },
                                        'OSQP': {
                                            'eps_abs':tol,
                                            'eps_rel':tol,
                                        },
                                        'SCS': {
                                            'eps':tol,
                                        },
                                        'CVXOPT': {
                                            'abstol':tol,
                                            'reltol':tol,
                                            'feastol':tol,
                                        },
                                        'COPT': {
                                            'AbsGap':tol,
                                        },
                                        'CLARABEL': {
                                            'tol_gap_abs':tol,
                                        },
                                    }
                                    solver_params = solver_params_[solver]


                                    # # SOCP
                                    # beta, xi, u, z, fo_value, time, p_time_mins, cg_dict = SOCP(X,y,tau,kappa,eps_sqrt=eps_sqrt, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)

                                    # # Error of the solution
                                    # # if socp.status == 'optimal':
                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # final_results.append(['SOCP',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp  ])
                                    # print(error_quad)
                                    # df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                    # df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_real_data2.csv', index=False)


                                    # Direct Relaxed Lasso
                                    # print(solver)
                                    # beta, z, fo_value, time, p_time_mins, cg_dict = L_LASSO(X,y,tau, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)

                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # final_results.append(['L_LASSO',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp   ])
                                    # print(error_quad)
                                    # df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                    # df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_real_data2.csv', index=False)



                                    # # Direct Relaxed Lasso, with scipy
                                    # print(solver)
                                    # scipy_solvers = ['L-BFGS-B']
                                    # for scipy_solver in scipy_solvers:
                                    #     print(scipy_solver)
                                    #     beta, z, fo_value, time, p_time_mins, cg_dict = L_LASSO_scipy(X,y,tau, solver=scipy_solver, solver_params={}, tol=tol)

                                    #     try:
                                    #         error_quad = np.linalg.norm(y - X @ beta) **2
                                    #     # else:
                                    #     except Exception as e:
                                    #         error_quad = None
                                    #     final_results.append(['L_LASSO_scipy',n_row,m_col, scipy_solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp ])
                                    #     print(error_quad)




                                    # # Elastic Net
                                    # if last_execution_times['SCIKIT_LASSO'] < models_time_limit:
                                    #     beta, z, fo_value, time, p_time_mins, cg_dict = SCIKIT_LASSO(X, y, tau, tol=tol)
                                    #     last_execution_times['SCIKIT_LASSO'] = time # update the last execution time

                                    #     # Error of the solution
                                    #     # if socp.status == 'optimal':
                                    #     try:
                                    #         error_quad = np.linalg.norm(y - X @ beta) **2
                                    #     # else:
                                    #     except Exception as e:
                                    #         error_quad = None
                                    #     final_results.append(['SCIKIT_LASSO', n_row,m_col, 'coordinate descent', error_quad_OLS, error_quad, fo_value, time, p_time_mins, cg_dict['k'],0,beta[beta > tol].size, 'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp   ])
                                    #     print(error_quad)

                                    #     df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                    #     df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_real_data2.csv', index=False)






                                    # # NO USAR
                                    # beta, xi, eta, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC2(
                                    #                         X, y, np.sqrt(tau*kappa), kappa, 
                                    #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                    #                         solver_params=solver_params, solver_verbose=solver_verbose,
                                    #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                    #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                    #                         v=max(int(m*0.012), 5), v0=max(int(m*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                    #                         canonic_random_initial_sol=True
                                    #                         )
                                    
                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # # final_results.append([m_col, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                    # final_results.append(['CG_LASSO_SOC2',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size, 'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp  ])
                                    # print(error_quad)






                                    # # beta, xi, u, z, xi_2, fo_value, time, p_time_mins, cg_dict = CG_SOC1_ElasticNet(
                                    # #                         X, y, tau, kappa, theta=theta, 
                                    # #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                    # #                         solver_params=solver_params, solver_verbose=solver_verbose,
                                    # #                        gi eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                    # #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                    # #                         v=max(int(m*0.012), 5), v0=max(int(m*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                    # #                         canonic_random_initial_sol=True
                                    # #                         )

                                    # # Error of the solution
                                    # # if socp.status == 'optimal':
                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # # final_results.append([m_col, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                    # final_results.append(['CG_LASSO_SOC1',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size, 'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta, tau_exp  ])
                                    # print(error_quad)





                                    # CG method
                                    # for init_type in ['random_canonical', 'random',  'correlation', 'ols']: # 'small_instance',
                                    if last_execution_times['CG_LASSO_SOC1_v2'] < models_time_limit:
                                        beta, xi, u,z, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC1_v2(
                                                                X, y, np.sqrt(tau*kappa), kappa, 
                                                                eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                                                solver_params=solver_params, solver_verbose=solver_verbose,
                                                                eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                                                solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                                                v=max(int(m*0.012), 5), v0=max(int(m*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                                                canonic_random_initial_sol=True,
                                                                unboundedness_policy=unboundedness_policy # 2025
                                                                )
                                        last_execution_times['CG_LASSO_SOC1_v2'] = time # update the last execution time
                                            
                                        try:
                                            error_quad = np.linalg.norm(y - X @ beta) **2
                                        # else:
                                        except Exception as e:
                                            error_quad = None
                                        # final_results.append([m_col, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                        final_results.append(['CG_LASSO_SOC1_v2',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, cg_dict['k'],cg_dict['k_v'],beta[beta > tol].size, 'converged', tau, kappa,np.sqrt(tau*kappa), alpha, theta, tau_exp ])
                                        # final_results.append(['CG_LASSO_SOC1_v2',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size, init_type, tau, kappa,np.sqrt(tau*kappa), alpha, theta, tau_exp ])
                                        print(error_quad)
                                        df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                        df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_real_data2.csv', index=False)


                                    # # CG method
                                    # # for init_type in ['random_canonical', 'random',  'correlation', 'ols']: # 'small_instance',
                                    # beta, xi, u,z, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC1(
                                    #                         X, y, np.sqrt(tau*kappa), kappa, 
                                    #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                    #                         solver_params=solver_params, solver_verbose=solver_verbose,
                                    #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                    #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                    #                         v=max(int(m*0.012), 5), v0=max(int(m*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                    #                         canonic_random_initial_sol=True
                                    #                         )
                                        
                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # # final_results.append([m_col, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                    # final_results.append(['CG_LASSO_SOC1',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, cg_dict['k'],cg_dict['k_v'],beta[beta > tol].size, 'converged', tau, kappa,np.sqrt(tau*kappa), alpha, theta, tau_exp ])
                                    # # final_results.append(['CG_LASSO_SOC1',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, 1,0,beta[beta > tol].size, init_type, tau, kappa,np.sqrt(tau*kappa), alpha, theta, tau_exp ])
                                    # print(error_quad)
                                    # df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                    # df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_real_data2.csv', index=False)




                                    # # for alpha_ in [1e-4, 1e-2, 1, 1e2, 1e4]:
                                    # # # alpha_ = 1
                                    # # # if True:

                                    # #     tau2 = tau/alpha_ # taul_tilda / alpha
                                    # #     kappa2 = alpha_ * tau # taul_tilda * alpha

                                    # beta, xi, u, z, fo_value, time, p_time_mins, cg_dict = CG_SOC1_upgrade(
                                    #                         X, y, tau, kappa, 
                                    #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                    #                         solver_params=solver_params, solver_verbose=solver_verbose,
                                    #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                    #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                    #                         v=max(int(m*0.012), 5), v0=max(int(m*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                    #                         canonic_random_initial_sol=True
                                    #                         )
                                    # try:
                                    #     error_quad = np.linalg.norm(y - X @ beta) **2
                                    # # else:
                                    # except Exception as e:
                                    #     error_quad = None
                                    # # final_results.append([m_col, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                    # final_results.append(['CG_SOC1_upgrade',n_row,m_col, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, cg_dict['k'],cg_dict['k_v'],beta[beta > tol].size, 'converged', tau, kappa,np.sqrt(tau*kappa), alpha_, theta, tau_exp ])
                                    # print(error_quad)





                                    # # Error of the solution
                                    # # Save final_results in a csv file
                                    # df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                    # df_final_results.to_csv(out_file, index=False) #'results/method-benchmark/main_methods_benchmark_synthetic_data2.csv', index=False)
                                    # # df_final_results.to_csv(out_file, index=False) #'results/benchmark/real-data-solver-benchmark.csv', index=False)
                                    # # df_final_results.to_csv(out_file, index=False) #'results/elastic-net/socp_CG_comparison_LASSO_soc2.csv', index=False)


























# # %%
# # df_final_results.to_csv(out_file, index=False) #'results/elastic-net/socp_CG_comparison.csv', index=False)

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df_results = pd.read_csv('results/elastic-net/socp_CG_comparison.csv')
# df_results.head()

# # plot the three models times as a function of tau/theta
# for m_col in df_results['m_col'].unique():
#     plt.figure(figsize=(8,5))
#     df_results_sub = df_results[df_results['m_col'] == m_col]
#     for model in df_results_sub['model'].unique():
#         plt.plot(df_results_sub[df_results_sub['model'] == model]['tau']/np.log(df_results_sub[df_results_sub['model'] == model]['theta']), 
#                 df_results_sub[df_results_sub['model'] == model]['time'], 'o',label=model)
#         # plt.plot(df_final_results[df_final_results['model'] == model]['m_col'], df_final_results[df_final_results['model'] == model]['time_process'], 'o--',label=model+' (process time)')
#     plt.legend()
#     plt.xlabel('tau/log(theta)')
#     plt.ylabel('time (mins)')
#     plt.title(f'Execution times as a function of tau (m={m_col})')
#     # plt.savefig('results/elastic-net/socp_CG_comparison_tau.pdf', format='pdf', dpi=300, bbox_inches='tight')
#     plt.show()


# # Now a heatmap of the same plot, with tau and theta in the axis. Consider that
# # the time, theta, and tau have the same length


# for m_col in df_results['m_col'].unique():
#     df_results_sub = df_results[df_results['m_col'] == m_col]
#     for model in df_results['model'].unique():
#         plt.figure(figsize=(8,5))
#         # df_sub_model = df_results_sub[df_results_sub['model'] == model]
#         # model = 'CG_SOC1_ElasticNet'
#         tau = df_results_sub[df_results_sub['model'] == model]['tau']
#         theta = np.log(df_results_sub[df_results_sub['model'] == model]['theta'])
#         time = df_results_sub[df_results_sub['model'] == model]['time']
#         plt.scatter(tau, theta, c=time, cmap='viridis', label=f'm={m_col}')
#         plt.colorbar()
#         plt.xlabel('tau')
#         plt.ylabel('log(theta)')
#         plt.title(f'Execution times as a function of tau and theta (m={m_col})\n {model}')
#         # plt.savefig('results/elastic-net/socp_CG_comparison_tau_theta.pdf', format='pdf', dpi=300, bbox_inches='tight')
#         plt.show()



# cp.installed_solvers()
# %%
                    
# # plot the three models times as a function of the sample size
# import matplotlib.pyplot as plt

# max_sample_size = 300
# plt.figure(figsize=(8,5))
# for model in df_final_results['model'].unique():
#     plt.plot(df_final_results[df_final_results['model'] == model]['m_col'], df_final_results[df_final_results['model'] == model]['time'], 'o--',label=model)
#     # plt.plot(df_final_results[df_final_results['model'] == model]['m_col'], df_final_results[df_final_results['model'] == model]['time_process'], 'o--',label=model+' (process time)')
# plt.legend()
# plt.xlabel('sample size (m)')
# plt.ylabel('time (mins)')
# plt.title('Execution times as a function of the sample size (m)')
# plt.savefig('results/elastic-net/socp_scikit_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()



# # %%

# # Plot the scikit-learn and CG_SOC1_ElasticNet fo_value percentage difference with respect to the SOCP fo_value, for each sample size
# plt.figure(figsize=(8,5))
# line_styles = ['o', 'd']
# for j,model in enumerate(['SCIKIT_ElasticNet', 'CG_SOC1_ElasticNet']):
#     socp_fo_values = df_final_results[df_final_results['model'] == 'SOCP']['fo_value'].to_numpy()
#     model_fo_values = df_final_results[df_final_results['model'] == model]['fo_value'].to_numpy()
#     percentage_difference = 100 * (model_fo_values - socp_fo_values) / socp_fo_values
#     plt.plot(df_final_results[df_final_results['model'] == model]['m_col'], percentage_difference, 
#              line_styles[j],label=model, markersize=6-j)
# plt.legend()
# plt.xlabel('sample size (m)')
# plt.ylabel('percentage difference (%)')
# plt.title('Percentage difference of the fo_value with respect to the SOCP fo_value')
# plt.savefig('results/elastic-net/socp_scikit_comparison_fo_value.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()
# %%
