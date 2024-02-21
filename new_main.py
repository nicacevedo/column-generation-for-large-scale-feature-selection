#%%
import cvxpy as cp
import numpy as np
import pandas as pd
import clarabel

# from time import time, process_time

import gurobipy as gp
from os import cpu_count

from models import CG_SOC1_upgrade, SOCP, SCIKIT_ElasticNet, CG_SOC1_ElasticNet, CG_LASSO_SOC1, CG_LASSO_SOC1_v2, CG_LASSO_SOC2, SCIKIT_LASSO, L_LASSO, L_LASSO_scipy
# import mosek as MSK

# Write a seond order cone program in Gurobi format.
# following notation https://www.cvxpy.org/examples/basic/socp.html :
# minimize   f'*x
# subject to ||A_i*x + b_i||_2 <= c_i'*x + d_i, i = 1, ..., m
#             F*x - g == 0

# class GUROBI_MODEL():

#     def __init__(self, solver_params={}):

#         self.solver_params = solver_params
#         self.model = gp.Model()

#     def add_variable(self, shape, vtype=gp.GRB.CONTINUOUS, positive=False, name='', **kwargs):
#         if positive:
#             return self.model.addMVar(shape, lb=int(0.0), name=name, **kwargs)
#         else:
#             return self.model.addMVar(shape, lb=-gp.GRB.INFINITY, name=name, **kwargs)

#     def add_constraint(self, constraint, name):
#         self.model.addConstr(constraint, name=name)

#     def add_SOC(self, A, b,  c, d, var, name):
#         """
#         Second order cone constraint in the form
#         ||A_i*x + b_i||_2 <= c_i'*x + d_i
#         """
#         self.model.addConstr(
#             ((A @ var + b) * (A @ var + b)).sum() <= (c @ var + d) * (c @ var + d), 
#             name=name
#             )

#     def add_objective(self, objective, name):
#         self.model.setObjective(objective, name=name)

#     def solve(self):
#         self.model.ModelSense = gp.GRB.MINIMIZE
#         self.model.optimize(**self.solver_params)
    


if __name__ == "__main__":
    # Sample of the data
    final_results = [['model','sample_size', 'solver', 'OLS_error_quad', 'error_quad', 'fo_value','time', 'time_process', 'n_betas','status', 'tau', 'kappa', 'lambda','alpha', 'theta']]
    for sample_size in [1000, 1500, 2000,2440]:#, 400, 500, 1000, 1500, 2000,2440]: # 100, 500, 1000,1500, 2000,  
        # 500, 1000, 1500, 2000, 2440
        X = pd.read_csv('real-data-treated/usa_n3522_m2440_yr2010_filled.csv', index_col=0, header=0)
        y = pd.read_csv('real-data/s&p_n3522_yr2010.csv', index_col=0, header=0)
        companies = X.columns.to_numpy()
        X.index = pd.to_datetime(X.index)
        y.index = pd.to_datetime(y.index)
        # only keep the same index of y and X (y is the target)
        common_index = y.index.intersection(X.index)
        X = X.loc[common_index].to_numpy()
        y = y.loc[common_index].to_numpy().T[0]
  
        # print(X.shape, y.shape)
        X = X[:,:sample_size]
        # y = y[:100]

        # Dimension of the problem
        n,m = X.shape

        # Standardize data
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        y = (y - np.mean(y))/np.std(y)

        # OLS solution
        beta_OLS = np.linalg.inv(X.T @ X) @ X.T @ y
        error_quad_OLS = np.linalg.norm(y - X @ beta_OLS) **2

        # Params
        eps_sqrt = 0
        init_tau = error_quad_OLS
       

        for tau_exp in [-3,-1,0,1,3]: # all previuos experiments were with tau_exp = 7
        #     for theta_exp in [-6,-3,0,3,6]:
            theta_exp = 1
            if True:
                tau_tilda = init_tau ** tau_exp  # tau = tau_tilda / alpha, and kappa = alpha **2 *  tau
                # 1/error bc it's a decimal number in most cases

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
                    time_limit = 90
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
                            # beta, xi, u, z, xi_2, fo_value, time, p_time_mins, cg_dict = SOCP(X,y,tau,kappa, theta=theta,eps_sqrt=eps_sqrt, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)

                            # # Error of the solution
                            # # if socp.status == 'optimal':
                            # try:
                            #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # else:
                            # except Exception as e:
                            #     error_quad = None
                            # final_results.append(['SOCP',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta])
                            # print(error_quad)



                            # # Direct Relaxed Lasso
                            # print(solver)
                            # beta, z, fo_value, time, p_time_mins, cg_dict = L_LASSO(X,y,tau, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)

                            # try:
                            #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # else:
                            # except Exception as e:
                            #     error_quad = None
                            # final_results.append(['L_LASSO',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta])
                            # print(error_quad)




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
                            #     final_results.append(['L_LASSO_scipy',sample_size, scipy_solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size,'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta])
                            #     print(error_quad)




                            # # Elastic Net
                            # beta, z, fo_value, time, p_time_mins, cg_dict = SCIKIT_LASSO(X, y, tau, tol=tol)

                            # # Error of the solution
                            # # if socp.status == 'optimal':
                            # try:
                            #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # else:
                            # except Exception as e:
                            #     error_quad = None
                            # final_results.append(['SCIKIT_LASSO', sample_size, 'coordinate descent', error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size, 'converged', tau, kappa, alpha, theta])
                            # print(error_quad)






                            # # # NO USAR
                            # # beta, xi, eta, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC2(
                            # #                         X, y, np.sqrt(tau*kappa), kappa, 
                            # #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                            # #                         solver_params=solver_params, solver_verbose=solver_verbose,
                            # #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                            # #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                            # #                         v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                            # #                         canonic_random_initial_sol=True
                            # #                         )
                            
                            # # try:
                            # #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # # else:
                            # # except Exception as e:
                            # #     error_quad = None
                            # # # final_results.append([sample_size, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                            # # final_results.append(['CG_LASSO_SOC2',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size, 'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta])
                            # # print(error_quad)






                            # # CG method
                            # beta, xi, eta, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC1(
                            #                         X, y, np.sqrt(tau*kappa), kappa, 
                            #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                            #                         solver_params=solver_params, solver_verbose=solver_verbose,
                            #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                            #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                            #                         v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                            #                         canonic_random_initial_sol=True
                            #                         )
                            


                            # # beta, xi, u, z, xi_2, fo_value, time, p_time_mins, cg_dict = CG_SOC1_ElasticNet(
                            # #                         X, y, tau, kappa, theta=theta, 
                            # #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                            # #                         solver_params=solver_params, solver_verbose=solver_verbose,
                            # #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                            # #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                            # #                         v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                            # #                         canonic_random_initial_sol=True
                            # #                         )

                            # # Error of the solution
                            # # if socp.status == 'optimal':
                            # try:
                            #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # else:
                            # except Exception as e:
                            #     error_quad = None
                            # # final_results.append([sample_size, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                            # final_results.append(['CG_LASSO_SOC1',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size, 'converged', tau, kappa, np.sqrt(tau*kappa),alpha, theta])
                            # print(error_quad)





                            # # CG method
                            # beta, xi, eta, fo_value, time, p_time_mins, cg_dict = CG_LASSO_SOC1_v2(
                            #                         X, y, np.sqrt(tau*kappa), kappa, 
                            #                         eps_soc2_sqrt=eps_sqrt, solver=solver, 
                            #                         solver_params=solver_params, solver_verbose=solver_verbose,
                            #                         eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                            #                         solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                            #                         v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                            #                         canonic_random_initial_sol=True
                            #                         )
                            
                            # try:
                            #     error_quad = np.linalg.norm(y - X @ beta) **2
                            # # else:
                            # except Exception as e:
                            #     error_quad = None
                            # # final_results.append([sample_size, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                            # final_results.append(['CG_LASSO_SOC1_v2',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size, 'converged', tau, kappa,np.sqrt(tau*kappa), alpha, theta])
                            # print(error_quad)






                            for alpha_ in [1e-5, 1, 1e5]:
                            # alpha_ = 1
                            # if True:

                                tau2 = tau/alpha_ # taul_tilda / alpha
                                kappa2 = alpha_ * tau # taul_tilda * alpha

                                beta, xi, u, z, fo_value, time, p_time_mins, cg_dict = CG_SOC1_upgrade(
                                                        X, y, tau2, kappa2, 
                                                        eps_soc2_sqrt=eps_sqrt, solver=solver, 
                                                        solver_params=solver_params, solver_verbose=solver_verbose,
                                                        eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                                        solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                                        v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                                        canonic_random_initial_sol=True
                                                        )
                                try:
                                    error_quad = np.linalg.norm(y - X @ beta) **2
                                # else:
                                except Exception as e:
                                    error_quad = None
                                # final_results.append([sample_size, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                                final_results.append(['CG_SOC1_upgrade',sample_size, solver, error_quad_OLS, error_quad, fo_value, time, p_time_mins, beta[beta > tol].size, 'converged', tau2, kappa2,np.sqrt(tau2*kappa2), alpha_, theta])
                                print(error_quad)





                                # Error of the solution
                                # Save final_results in a csv file
                                df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                                # df_final_results.to_csv('results/benchmark/real-data-solver-benchmark.csv', index=False)
                                df_final_results.to_csv('results/elastic-net/socp_CG_comparison_LASSO.csv', index=False)


# # %%
# # df_final_results.to_csv('results/elastic-net/socp_CG_comparison.csv', index=False)

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df_results = pd.read_csv('results/elastic-net/socp_CG_comparison.csv')
# df_results.head()

# # plot the three models times as a function of tau/theta
# for sample_size in df_results['sample_size'].unique():
#     plt.figure(figsize=(8,5))
#     df_results_sub = df_results[df_results['sample_size'] == sample_size]
#     for model in df_results_sub['model'].unique():
#         plt.plot(df_results_sub[df_results_sub['model'] == model]['tau']/np.log(df_results_sub[df_results_sub['model'] == model]['theta']), 
#                 df_results_sub[df_results_sub['model'] == model]['time'], 'o',label=model)
#         # plt.plot(df_final_results[df_final_results['model'] == model]['sample_size'], df_final_results[df_final_results['model'] == model]['time_process'], 'o--',label=model+' (process time)')
#     plt.legend()
#     plt.xlabel('tau/log(theta)')
#     plt.ylabel('time (mins)')
#     plt.title(f'Execution times as a function of tau (m={sample_size})')
#     # plt.savefig('results/elastic-net/socp_CG_comparison_tau.pdf', format='pdf', dpi=300, bbox_inches='tight')
#     plt.show()


# # Now a heatmap of the same plot, with tau and theta in the axis. Consider that
# # the time, theta, and tau have the same length


# for sample_size in df_results['sample_size'].unique():
#     df_results_sub = df_results[df_results['sample_size'] == sample_size]
#     for model in df_results['model'].unique():
#         plt.figure(figsize=(8,5))
#         # df_sub_model = df_results_sub[df_results_sub['model'] == model]
#         # model = 'CG_SOC1_ElasticNet'
#         tau = df_results_sub[df_results_sub['model'] == model]['tau']
#         theta = np.log(df_results_sub[df_results_sub['model'] == model]['theta'])
#         time = df_results_sub[df_results_sub['model'] == model]['time']
#         plt.scatter(tau, theta, c=time, cmap='viridis', label=f'm={sample_size}')
#         plt.colorbar()
#         plt.xlabel('tau')
#         plt.ylabel('log(theta)')
#         plt.title(f'Execution times as a function of tau and theta (m={sample_size})\n {model}')
#         # plt.savefig('results/elastic-net/socp_CG_comparison_tau_theta.pdf', format='pdf', dpi=300, bbox_inches='tight')
#         plt.show()



# cp.installed_solvers()
# %%
                    
# # plot the three models times as a function of the sample size
# import matplotlib.pyplot as plt

# max_sample_size = 300
# plt.figure(figsize=(8,5))
# for model in df_final_results['model'].unique():
#     plt.plot(df_final_results[df_final_results['model'] == model]['sample_size'], df_final_results[df_final_results['model'] == model]['time'], 'o--',label=model)
#     # plt.plot(df_final_results[df_final_results['model'] == model]['sample_size'], df_final_results[df_final_results['model'] == model]['time_process'], 'o--',label=model+' (process time)')
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
#     plt.plot(df_final_results[df_final_results['model'] == model]['sample_size'], percentage_difference, 
#              line_styles[j],label=model, markersize=6-j)
# plt.legend()
# plt.xlabel('sample size (m)')
# plt.ylabel('percentage difference (%)')
# plt.title('Percentage difference of the fo_value with respect to the SOCP fo_value')
# plt.savefig('results/elastic-net/socp_scikit_comparison_fo_value.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()
# %%
