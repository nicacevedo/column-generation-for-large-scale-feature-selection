#%%
import cvxpy as cp
import numpy as np
import pandas as pd
import clarabel

# from time import time, process_time

import gurobipy as gp
from os import cpu_count

from models import CG_SOC1_upgrade, SOCP
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
    final_results = [['sample_size', 'solver', 'error_quad', 'fo_value','time', 'time_process', 'status', 'tau', 'kappa', 'alpha']]
    for sample_size in [100, 500, 1000,1500, 2000,2440]: # 100, 500, 1000,1500, 2000,  

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
        tau_tilda = init_tau ** 7  # tau = tau_tilda / alpha, and kappa = alpha **2 *  tau
        for alpha_ in [.2, .4, .6, .8,  1]:
            alpha = 1/alpha_
            # tau = (1 - ratio) * tau_tilda
            tau = tau_tilda / alpha
            # kappa =  
            kappa = alpha ** 2 * tau
            print('tau2', tau, 'kappa', kappa, 'alpha', alpha)
            print('sqrt(tau*kappa)', np.sqrt(tau*kappa), 'sqrt(tau_tilda**2)', tau_tilda)

            solver_verbose = False
            solvers = ['MOSEK'] # ,  'CVXOPT', 'MOSEK', 'GUROBI', 'ECOS', 'ECOS_BB', 'SCS'
            add_constant = False
            time_limit = 90
            for solver in solvers:
                for i in range(10):
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


                    # Model 
                    results = CG_SOC1_upgrade(
                                            X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                            eps_soc2_sqrt_L=eps_sqrt, add_constant=add_constant, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
                                            solve_dual_directly=False, cg_lambda_tol=tol, cg_residuals_tol=tol,
                                            v=max(int(n*0.012), 5), v0=max(int(n*0.012), 5), time_limit=time_limit, dummy_condition=False,
                                            canonic_random_initial_sol=True
                                            )

                    # results = SOCP(X,y,tau,kappa, eps_sqrt=eps_sqrt, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                    
                    # results =  beta.value, xi.value, u.value, z.value, socp.value, (t1-t0)/60, (t1p-t0p)/60, {}
                    # print(results)
                    # results = beta, xi, u, z, fo_value, time, p_time_mins, cg_dict
                    beta, xi, u, z, fo_value, time, p_time_mins, cg_dict = results

                    # Error of the solution
                    # if socp.status == 'optimal':
                    try:
                        error_quad = np.linalg.norm(y - X @ beta) **2
                    # else:
                    except Exception as e:
                        error_quad = None
                    # final_results.append([sample_size, solver, error_quad, 'optimal', (t1-t0)/60, (t1p-t0p)/60, socp.status])
                    final_results.append([sample_size, solver, error_quad, fo_value, time, p_time_mins, 'converged', tau, kappa, alpha])
                    print(error_quad)

                    # Save final_results in a csv file
                    df_final_results = pd.DataFrame(final_results[1:], columns=final_results[0])
                    df_final_results.to_csv('results/benchmark/real-data-solver-benchmark.csv', index=False)



# %%
cp.installed_solvers()
# %%
