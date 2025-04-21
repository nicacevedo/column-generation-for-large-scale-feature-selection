from models import MIQP, SOCP, L_LASSO, SCIKIT_LASSO, \
     CG_SOC1_upgrade, CG_SOC2,  CG_SOC1_SOC2_upgrade, MIP_R, \
        CG_LASSO_SOC1_v2, CG_LASSO_SOC1

import cvxpy as cp

import os.path
from os import cpu_count

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

def main():
    
    # Size of the data (last real: (83976, 447))
    n_cols =[(i+1)*100 for i in range(0,5)]#[1699]#[200]#[(i+1)*1000 for i in range(2,5)] # [(i+1)*100 for i in range(10)] + 
    n_rows = [1000]#[4884]#[10000]
    p_error = 20

    # Beta to be considered as nonzero
    pos_tol = 1e-6

    # Solver tolerance
    solver_gap_tol = 1e-6
    solver_params = {
        'MOSEK': {
        # Relative complementarity gap tolerance used by the interior-point optimizer for conic problems.
        # 'MSK_DPAR_INTPNT_CO_TOL_MU_RED':solver_gap_tol, 
        # Relative gap termination tolerance used by the interior-point optimizer for conic problems.
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':solver_gap_tol,
        # Primal feasibility tolerance used by the interior-point optimizer for conic problems.
        # 'MSK_DPAR_INTPNT_CO_TOL_PFEAS':solver_gap_tol,
        # Dual feasibility tolerance used by the interior-point optimizer for conic problems.
        # 'MSK_DPAR_INTPNT_CO_TOL_DFEAS':solver_gap_tol,
        'MSK_IPAR_NUM_THREADS':cpu_count(),
        },
        'GUROBI': {
            'BarConvTol':solver_gap_tol,
            'BarQCPConvTol':solver_gap_tol
        },
        'ECOS': {
            'abstol':solver_gap_tol,
            'reltol':solver_gap_tol,
            'feastol':solver_gap_tol,
        },
        'ECOS_BB': {
            'abstol':solver_gap_tol,
            'reltol':solver_gap_tol,
            'feastol':solver_gap_tol,
        },
        'OSQP': {
            'eps_abs':solver_gap_tol,
            'eps_rel':solver_gap_tol,
        },
        'SCS': {
            'eps':solver_gap_tol,
        },
        'CVXOPT': {
            'abstol':solver_gap_tol,
            'reltol':solver_gap_tol,
            'feastol':solver_gap_tol,
        },
        'COPT': {
            'AbsGap':solver_gap_tol,
        },
        'CLARABEL': {
            'tol_gap_abs':solver_gap_tol,
        },
    }


    # Cone noise for faster convergence (Not working well)
    eps_sqrt = 0#1e-8

    # Solver: MOSEK, GUROBI
    solver = cp.MOSEK

    # Verbose
    solver_verbose = False

    # Exponent rate of smoothing m in tildas
    m_exp_rate = 0#.4
           
    # Number of solutions to include in each iteration
    # v = int(n_col*0.012)   # number of solutions to include in each iteration
    v0 = 5

    # CG time limit
    time_limit = 60 

    # Optimization method
    opt_methods = [
        CG_LASSO_SOC1, CG_LASSO_SOC1_v2, # LASSO CG
        CG_SOC1_upgrade, CG_SOC2, CG_SOC1_SOC2_upgrade, # CONIC CG
        SOCP, MIQP, MIP_R, L_LASSO,  # MONOLITIC
        SCIKIT_LASSO, # SCIKIT
    #     'CG_LASSO_SOC1', 'CG_LASSO_SOC1_v2', # LASSO CG
    #    'CG_SOC1_upgrade', 'CG_SOC2', 'CG_SOC1_SOC2_upgrade', # CONIC CG   
    #     'SOCP', 'MIQP', 'MIP_R', 'L_LASSO',  # MONOLITIC
    #     'SCIKIT_LASSO', # SCIKIT
                   ] 

    # Standardize and normalize data
    stand_norm = True
    # add constant to data
    add_constant_to_X = False#True

    # data_type = 'real' or 'synthetic'
    data_types = ['synthetic-data-error', 'real-data'] # treated: deleted rows with NaN values
    data_type = data_types[1]

    # Save convergence info
    save_conv_info = False

    # Dummy condition of lagrangian
    dummy_condition = False

    # add constant to initial solution (v_0 + 1)
    add_constant_initial_sol, canonic_random_initial_sol = (False, True)

    sum_1_comb, pos_linear_comb = False, False

    # # OLD RESULTS
    # if 'CG_' in opt_method:
    #     results = [[
    #         'n', 'm', 'opt_method', 'tau', 'kappa', 'time_mins', 'p_time_mins', 'fo', 'error_OLS', 'error_model', 'n_betas',
    #         'k_iter', 'k_iter_v', 'v', 'v0', 'one_v0', 'canon_v0', 'solver', 'solver_gap_tol', 'm_exp_rate', 'eps_sqrt', 'pos_tol', 'time_limit', 'dummy_condition',
    #         'save_conv_info', 'sum_1_comb', 'pos_linear_comb'
    #         ]] # PASAR A VECTORES EN OTRO LADO: 'betas_OLS', 'betas_model'
    # elif opt_method in ['MIQP', 'SOCP', 'L_LASSO', 'SCIKIT_LASSO']:
    #     results = [[
    #         'n', 'm', 'opt_method', 'tau', 'kappa', 'time_mins', 'p_time_mins', 'fo', 'error_OLS', 'error_model', 'n_betas',
    #         'solver', 'solver_gap_tol', 'm_exp_rate', 'eps_sqrt', 'pos_tol'
    #         ]] # , 'n_noise_in_betas' # , 'error_OLS_pred', 'error_model_pred'
    # NEW RESULTS
    results = [[
        'n', 'm', 'opt_method', 'tau', 'kappa', 'alpha', 'time_mins', 'p_time_mins', 'fo', 'error_OLS', 'error_model', 'n_betas',
        'k_iter', 'k_iter_v', 'v', 'v0', 'one_v0', 'canon_v0', 'solver', 'solver_gap_tol', 'm_exp_rate', 'eps_sqrt', 'pos_tol', 'time_limit', 'dummy_condition',
        'save_conv_info', 'sum_1_comb', 'pos_linear_comb'
        ]] 
    
        

    betas_results = []

    for n_col in n_cols:
        v = max(int(n_col*0.012), 5) #max(int(n_col*0.03), 5) 
        # v0 = max(int(n_col*0.02), 5)
        for n_row in n_rows:
            print("\n","-"*10,"\n",f"(n={n_row},m={n_col})")

            if data_type == 'real-data':
                X = pd.read_csv('real-data-treated/usa_n3522_m2440_yr2010_filled.csv', index_col=0, header=0)
                y = pd.read_csv('real-data/s&p_n3522_yr2010.csv', index_col=0, header=0)

                # companies = X.columns.to_numpy()
                X.index = pd.to_datetime(X.index)
                y.index = pd.to_datetime(y.index)
                # only keep the same index of y and X (y is the target)
                common_index = y.index.intersection(X.index)
                X = X.loc[common_index].to_numpy()
                y = y.loc[common_index].to_numpy().T[0]
        
                # print(X.shape, y.shape)
                X = X[:n_row,:n_col]
                y = y[:n_row]
            else:
                # Base de datos (Parámetros del PPL)
                data_type_name = data_type.split('-')[0]
                y = np.loadtxt(os.path.dirname(__file__) + f'/../{data_type}/{data_type_name}_y_{n_col}_{n_row}_e{p_error}.csv'.format(), delimiter=',')
                X = np.loadtxt(os.path.dirname(__file__) + f'/../{data_type}/{data_type_name}_X_{n_col}_{n_row}_e{p_error}.csv', delimiter=',')
        
            # y_data = pd.read_pickle(os.path.dirname(__file__) + f'/../Bases/{data_type}_y_{n_col}_{n_row}.pkl').to_numpy().reshape(-1)
            # X_data = pd.read_pickle(os.path.dirname(__file__) + f'/../Bases/{data_type}_X_{n_col}_{n_row}.pkl').to_numpy()

            print("Base de datos leída")

            # X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0, random_state=42)

            # Standardize data
            if stand_norm: 
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                y = (y - y.mean()) / y.std() # Only neccesary for gradient methods??

                # test
                # X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
                # y_test = (y_test - y_test.mean()) / y_test.std()

            # add a constant column to X
            if add_constant_to_X: 
                X = np.hstack((np.ones((X.shape[0], 1)), X))
                # X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
            

            N,M = X.shape

            # Solución de regresión clásica
            try:
                beta_OLS = np.linalg.inv(X.T @ X) @ (X.T @ y)
            except Exception as e:
                print(e)
                try:
                    # pseudo inverse solution of OLS
                    beta_OLS = np.linalg.pinv(X) @ y # This one seems to perform even better, but it is slower
                except Exception as e:
                    print(e)
                    # second alternative of pseudo inverse (Not sure if this is correct)
                    beta_OLS = np.linalg.pinv(X.T @ X) @ (X.T @ y)

            # Quadratic error in OLS: ||(y_hat - y)||^2
            error_quad_OLS = (X @ beta_OLS - y) @ (X @ beta_OLS - y)
            # error_quad_OLS_test = (X_test @ beta_OLS - y_test) @ (X_test @ beta_OLS - y_test)


            # CONTINUE FROM HERE

            # Iterate over the optimization methods
            for opt_method in [CG_LASSO_SOC1_v2, # LASSO CG
        CG_SOC1_upgrade, # CONIC CG
        SOCP, L_LASSO,  # MONOLITIC
        SCIKIT_LASSO]:  #opt_methods:
                # Iterate over the solvers
                for solver in ['MOSEK']:

                    # General parameters
                    general_params = {
                        'solver':[solver],
                        'solver_params':[solver_params[solver]],
                        'solver_verbose':[False],
                    }

                    # Conic parameters
                    if 'SOC' in opt_method.__name__ and not 'LASSO' in opt_method.__name__:
                        soc_params = {
                            'eps_sqrt':[0], 
                            'eps_sqrt_L':[0], 
                        }
                        general_params = general_params | soc_params

                    # Method parameters
                    if 'CG' in opt_method.__name__:
                        method_params = {
                            'save_conv_info':[False], 
                            'solve_dual_directly':[False],
                            # combination of solutions (convex, conic, affine or linear) 
                            'sum_1_comb':[False], 
                            'pos_linear_comb':[False],
                            # CG convergence 
                            'cg_lambda_tol':[solver_gap_tol], 
                            'cg_residuals_tol':[solver_gap_tol],
                            'time_limit':[None],
                            # initial solution
                            'add_constant':[True], 
                            'canonic_random_initial_sol':[True],
                            'v0':[5],
                            # synthetic solutions
                            'v':[5],
                            'dummy_condition':[False],
                        }

                        general_params = general_params | method_params


                    # Every combination of method parameters
                    keys, values = zip(*general_params.items())
                    for bundle in product(*values):
                        print(bundle)
                        if 'CG' in opt_method.__name__ and not 'LASSO' in opt_method.__name__:
                            solver, solver_params, solver_verbose, eps_sqrt, eps_sqrt_L, save_conv_info, solve_dual_directly, sum_1_comb, pos_linear_comb, cg_lambda_tol, cg_residuals_tol, time_limit, add_constant_initial_sol, canonic_random_initial_sol, v0, v, dummy_condition = bundle
                        elif 'CG' in opt_method.__name__: 
                            solver, solver_params, solver_verbose, save_conv_info, solve_dual_directly, sum_1_comb, pos_linear_comb, cg_lambda_tol, cg_residuals_tol, time_limit, add_constant_initial_sol, canonic_random_initial_sol, v0, v, dummy_condition = bundle
                        elif 'SOC' in opt_method.__name__ and not 'LASSO' in opt_method.__name__: 
                            solver, solver_params, solver_verbose, eps_sqrt, eps_sqrt_L = bundle
                        else:
                            solver, solver_params, solver_verbose = bundle
                        model_params = dict(zip(keys, bundle))
                        # solver, solver_params, solver_verbose
                        # model_params = dict(zip(keys1, bundle1))
                        # solver, solver_params, solver_verbose, eps_sqrt, eps_sqrt_L = model_params['solver'], model_params['solver_params'], model_params['solver_verbose'], model_params['eps_sqrt'], model_params['eps_sqrt_L']
                        # # model_params = dict(zip(keys1, bundle1)) | general_params
                        
                

                

                

                # for opt_method in ['CG_SOC1_upgrade', 'CG_SOC1_naive', 'CG_SOC2_naive', 'CG_SOC2_basic', 'CG_SOC2']: # , # 'CG_SOC2', 'CG_SOC1_SOC2', 'CG_SOC1_upgrade', 'SOCP' 'CG_SOC1_dummy',
                #     if opt_method == 'CG_SOC1_upgrade':
                #         # Time limit
                #         time_limit = None
                #     elif SOC1_time_limit is not None:
                #         time_limit = SOC1_time_limit
                # v0 = max(int(n_col*0.012),5)###
                # if True:
                # for v in range(1, 10, 2):
                # for v in range(int(n_col*0.025)//5,int(n_col*0.025)+2, int(n_col*0.025)//5):
                # for v0 in range(1,int(n_col*0.25)+2, int(n_col*0.25)//5):


                # m_exp_rate = 20/100
                
                # tau_0 = error_quad_OLS/M**0.5
                # tau_f = error_quad_OLS
                # for tau in np.arange(tau_0, tau_f, (tau_f-tau_0)/10):
                #     tau, kappa = tau, tau

                # for pair in [ (False, True), (False, False), (True,False), (True,True)]:
                #     add_constant_initial_sol, canonic_random_initial_sol = pair
                # #     sum_1_comb, pos_linear_comb = pair 

                # for i in range(1):
                #     # if True:


                # for eps_sqrt_e6 in range(0, 1, 1):

                # Parametro a mover
                # if True:
                # for v0 in range(1,int(n_col*0.1)+2,int(n_col*0.1)//10):
                #     print('v0:', v0, 'soluciones iniciales')
                #     for add_constant_initial_sol in [True, False]:
                #         print('add_constant_initial_sol:', add_constant_initial_sol)
                        # add_constant_initial_sol = False

                # for v in range(max(int(n_col*0.0075), 5), int(n_col*0.03)+1, int(n_col*0.03)//10):
                # for v in range(12, 37, 2):      


                # for dummy_condition in [True]:

                # for dummy_condition in [ False]: # True,
                    # for opt_method in ['CG_SOC1_upgrade', 'SCIKIT_LASSO', 'L_LASSO', 'SOCP']:
                #     for alpha in [2**(a//2) for a in range(0,11,5)]:
                # if True:
                        for m_exp_rate in [-5,-3,-1,0,1,3]: 


                            # eps_sqrt = 0#eps_sqrt_e6*1e-6
                            # eps_sqrt_L = 0 # eps_sqrt

                            # Punishment for the coefficients (forgetting about tildas)
                            tau = error_quad_OLS/M**m_exp_rate #np.sqrt(error_quad_OLS)/(M**m_exp_rate) #(M**m_exp_rate)
                            # tau = np.sqrt(error_quad_OLS)/M*m_exp_rate_pond
                            kappa = tau

                            alpha = 1
                            
                            tau = tau/alpha
                            kappa = alpha*kappa

                            # Optimization method
                            try:
                                if 'CG' in opt_method.__name__:
                                    beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = opt_method(
                                        X, y, tau, kappa, **model_params
                                        # X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                        # eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                        # solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                        # v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition
                                        )
                                # if opt_method == 'CG_SOC1_upgrade':
                                #     beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = CG_SOC1_upgrade(
                                #         X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                #         eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                #         solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                #         v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition,
                                #         canonic_random_initial_sol=canonic_random_initial_sol
                                #         )
                                # elif opt_method == 'CG_SOC2':
                                #     beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = CG_SOC2(
                                #         X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                #         eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                #         solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                #         v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition
                                #         )
                                # elif opt_method == 'CG_SOC1_SOC2_upgrade':
                                #     beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = CG_SOC1_SOC2_upgrade(
                                #         X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                #         eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                #         solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                #         v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition
                                #         )
                                # elif opt_method == 'CG_LASSO_SOC1':
                                #     beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = CG_LASSO_SOC1(
                                #         X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                #         eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                #         solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                #         v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition
                                #         )
                                # elif opt_method == 'CG_LASSO_SOC1_v2':
                                #     beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = CG_LASSO_SOC1_v2(
                                #         X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose, 
                                #         eps_soc2_sqrt_L=eps_sqrt_L, add_constant=add_constant_initial_sol, save_conv_info=save_conv_info, sum_1_comb=sum_1_comb, pos_linear_comb=pos_linear_comb,
                                #         solve_dual_directly=False, cg_lambda_tol=solver_gap_tol, cg_residuals_tol=solver_gap_tol,
                                #         v=v, v0=v0, time_limit=time_limit, dummy_condition=dummy_condition
                                #         )

                                elif opt_method.__name__ == 'SOCP':
                                    beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = SOCP(
                                        X, y, tau, kappa, **model_params
                                        # X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose
                                        )
                                elif opt_method.__name__ == 'MIP_R':
                                    beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = MIP_R(
                                        X, y, tau, kappa, **model_params
                                        # X, y, tau, kappa, eps_sqrt, solver, solver_params, solver_verbose
                                        )
                                # elif opt_method.__name__ == 'MIQP':
                                #     beta, z, fo_value, time_mins, p_time_mins, cg_dict = MIQP(X, y, tau, solver, solver_params, solver_verbose)
                                elif opt_method.__name__ == 'L_LASSO':
                                    beta, z, fo_value, time_mins, p_time_mins, cg_dict = L_LASSO(
                                        X, y, tau, kappa, **model_params
                                        # X, y, tau, solver, solver_params, solver_verbose
                                        )
                                elif opt_method.__name__ == 'SCIKIT_LASSO':
                                    beta, xi, u, z, fo_value, time_mins, p_time_mins, cg_dict = SCIKIT_LASSO(
                                        X, y, tau, kappa,lasso_type='Lasso', tol=solver_gap_tol, **model_params
                                        # X, y, tau, solver, solver_params, solver_verbose, 
                                        )
                                

                            except Exception as e:
                                print(e)
                                continue
                            # Print parameters # , error_quad_OLS_pred={error_quad_OLS_pred}
                            print(f"v={v}, m_exp_rate={m_exp_rate}, error_quad_OLS={error_quad_OLS}, fo_value={fo_value}, time_mins={time_mins}, p_time_mins={p_time_mins}")

                            # if not solution
                            if beta is None:
                                continue

                            # # Print gamma (z / u)
                            # print('tau', tau, 'kappa', kappa)
                            # print('kappa/tau', kappa/tau)
                            # print("gamma:", z/u)
                            # print("beta/z", beta/z)


                            # Second validation
                            # print(X.shape)
                            # print(beta)
                            # print(beta.shape)
                            y_val = X @ beta

                            # Second prediction
                            # y_pred = X_test @ beta 
                            
                            # # Plot real vs predicted
                            # sample_index = np.random.choice(y.size, 200, replace=False)
                            # plt.figure(figsize=(6,6))
                            # if y.size <= 200:
                            #     plt.scatter(y, y_val, facecolors='none', edgecolors='C0')
                            # else:
                            #     plt.scatter(y[sample_index], y_val[sample_index], facecolors='none', edgecolors='C0')
                            # plt.scatter(y, y_val, facecolors='none', edgecolors='C0')
                            # plt.title(f"y v/s y_hat (Regular)\nn_col={M}, n_row={N}")
                            # plt.xlabel("y")
                            # plt.ylabel("y_hat")
                            # plt.show()

                            # # # Barplot of betas
                            # # plt.figure(figsize=(6,6))
                            # # plt.bar(np.arange(M), beta, color='C0')
                            # # plt.title(f"Betas (Regular)\nn_col={M}, n_row={N}")
                            # # plt.xlabel("Betas")
                            # # plt.ylabel("Value")
                            # # plt.show()


                            if 'CG' in opt_method.__name__.__name__ and save_conv_info:   
                                # Plot convergence values evolution
                                master_values = cg_dict['master_values']
                                print("master_values", master_values)
                                lagrangian_values = cg_dict['lagrangian_values']
                                if len(lagrangian_values) < len(master_values):
                                    lagrangian_values.append(None)
                                print("lagrangian_values", lagrangian_values)
                                dual_values = cg_dict['dual_values']
                                master_times = cg_dict['master_times']
                                print("master times", master_times)
                                lagrangian_times = cg_dict['lagrangian_times']
                                if len(lagrangian_times) < len(master_times):
                                    lagrangian_times.append(None)
                                print("lagrangian times", lagrangian_times)
                                dual_times = cg_dict['dual_times']
                                plt.figure(figsize=(8,5))
                                plt.plot(np.arange(1,len(master_times)+1), master_values, '--o', c='C0', label="Master problem")
                                # plt.plot(np.arange(len(dual_values)), dual_values, 'x', c='C2', label="Problema dual (maestro)")
                                plt.plot(np.arange(1,len(master_times)+1), lagrangian_values, '--o', c='C1', label="Lagrange relaxation")
                                plt.legend()
                                # plt.title("Valores objetivo óptimos por iteración del método $CG - L_{C1}$" + f"\n(n={n_row}, m={n_col})")
                                # plt.xlabel("Iteración (k)")
                                # plt.ylabel("Valor objetivo óptimo")
                                # Translate to english the labels and title
                                plt.title("$CG - L_{C1}$")
                                # plt.title("Optimal objective values by iteration of the $CG - L_{C1}$ method" + f"\n(n={n_row}, m={n_col})")
                                plt.xlabel("Iteration (k)")
                                plt.ylabel("Optimal objective value")
                                # Only consider integers on x axis
                                # plt.xticks(np.arange(len(master_values)))
                                # plt.savefig(os.path.dirname(__file__) + f'/../Analysis/plots/results2/{opt_method.__name__}_n{n_row}_m{n_col}_m_rate{m_exp_rate}_dummy_cond{dummy_condition}_conv_eng.pdf')
                                plt.show()

                                # # Display a dataframe with the values of convergence
                                # display(pd.DataFrame({'master_values':master_values, 'lagrangian_values':lagrangian_values},
                                #                      index=np.arange(len(master_values))))


                                # Plot solver times evolution
                                plt.figure(figsize=(8,5))
                                plt.plot(np.arange(1,len(master_times)+1), master_times, '--o', c='C0', label="Master problem")
                                # plt.plot(np.arange(len(dual_times)), dual_times, 'x', c='C2', label="Problema dual (maestro)")
                                plt.plot(np.arange(1,len(master_times)+1), lagrangian_times, '--o', c='C1', label="Lagrange relaxation")
                                plt.legend()
                                # plt.title("Tiempos de ejecución por iteración del método $CG - L_{C1}$" + f"\n(n={n_row}, m={n_col})")
                                # plt.xlabel("Iteración (k)")
                                # plt.ylabel("Tiempo de ejecución(minutos)")
                                # Translate to english the labels and title
                                plt.title("$CG - L_{C1}$")
                                # plt.title("Execution times by iteration of the $CG - L_{C1}$ method" + f"\n(n={n_row}, m={n_col})")
                                plt.xlabel("Iteration (k)")
                                plt.ylabel("Execution time (minutes)")
                                # Only consider integers on x axis
                                # plt.xticks(np.arange(len(master_times)))
                                # plt.savefig(os.path.dirname(__file__) + f'/../Analysis/plots/results2/{opt_method.__name__}_n{n_row}_m{n_col}_m_rate{m_exp_rate}_dummy_cond{dummy_condition}_time_eng.pdf')
                                # plt.savefig(os.path.dirname(__file__) + f'/../Analysis/plots/results2/{opt_method.__name__}_n{n_row}_m{n_col}_v{v}_v0{v0}_m_exp_rate{m_exp_rate}_dummy_condition{dummy_condition}_times.pdf')
                                plt.show()

                                # # Display a dataframe with the values of convergence
                                # display(pd.DataFrame({'master_times':master_times, 'lagrangian_times':lagrangian_times},
                                #                         index=np.arange(len(master_times))))

                            # if opt_method.__name__ == 'CG_SOC1_naive':
                            # if opt_method.__name__ == 'CG_SOC1_upgrade':
                            #     SOC1_time_limit = time_mins*20 


                            
                            # Error
                            error_val = (y_val - y) @ (y_val - y)
                            # error_pred = (y_pred - y_test) @ (y_pred - y_test)
                            print("error val OLS:", error_quad_OLS)
                            print(f"error val {opt_method.__name__}:", error_val)
                            print("The optimal value is:", fo_value) 
                            print(f"error val {opt_method.__name__} sqrt:", np.sqrt(error_val))
                            print("#Betas originales", beta_OLS[beta_OLS!=0].size)
                            print("#Betas sobrev.:", np.sum(beta > pos_tol), '(', 100*np.sum(beta > pos_tol)/M,'%)')
                            # n_noise_in_betas = np.sum([x in noise_columns for x in np.where(beta > pos_tol)[0]])
                            # print("#Betas sobrev. in noise columns:", n_noise_in_betas, '(', 100*n_noise_in_betas/ np.sum(beta > pos_tol),'%)')
                            print("Betas sobrev.:", np.where(beta > pos_tol)[0])
                            # print("Values of betas:", beta[beta > pos_tol])

                            # Add to results dataframe
                            # results = results.append({'n': N, 'm': M, 'time_mins': time_mins, 'p_time_mins': p_time_mins, 'error_OLS': error_quad_OLS, 'error_OLS_pred': error_quad_OLS_pred, 'error_model': error_val, 'error_model_pred': error_pred, 'betas_OLS': beta_OLS, 'betas_model': (beta > pos_tol).astype(int)}, ignore_index=True)
                            if 'CG_' in opt_method.__name__:
                                results.append([
                                    N, M, opt_method.__name__, tau, kappa, alpha, time_mins, p_time_mins, fo_value, error_quad_OLS, error_val, np.sum(beta > pos_tol),
                                    cg_dict['k'], cg_dict['k_v'], v, v0, add_constant_initial_sol, canonic_random_initial_sol, solver, solver_gap_tol, m_exp_rate, eps_sqrt, pos_tol, time_limit, dummy_condition,
                                    save_conv_info,  sum_1_comb, pos_linear_comb
                                    ]) # , n_noise_in_betas , error_quad_OLS_pred, error_pred
                            elif opt_method.__name__ in ['MIQP','SOCP', 'L_LASSO', 'SCIKIT_LASSO']:
                                results.append([
                                    N, M, opt_method.__name__, tau, kappa, alpha, time_mins, p_time_mins, fo_value, error_quad_OLS, error_val, np.sum(beta > pos_tol),
                                    None, None, None, None, None, None, solver, solver_gap_tol, m_exp_rate, eps_sqrt, pos_tol, None, None,
                                    None, None, None
                                    ]) # , n_noise_in_betas, error_quad_OLS_pred, error_pred
                                
                            # # force file name    (WARNING: only when looping opt_methods)
                            # opt_method.__name__ = 'CG_all_borrar'}

                            # Save results
                            final_results = pd.DataFrame(results[1:], columns=results[0])
                            final_results.to_csv(os.path.dirname(__file__) + '/../results/methods-benchmark/methods_{}_{}.csv'.format(data_type, solver), index=False)

                            # Append beta to results as a list and save betas_results as a dataframe to a file
                            betas_results.append(beta.tolist())
                            pd.DataFrame(betas_results).to_csv(os.path.dirname(__file__) + '/../results/method-benchmark/betas/methods_{}_{}_betas.csv'.format(data_type, solver), index=False)


                    # np.savetxt(os.path.dirname(__file__) + '/../Results/betas/main_opt_method{}_{}_{}_betas.csv'.format(opt_method.__name__, data_type, solver), betas_results, delimiter=',')
    # final_results.to_csv(os.path.dirname(__file__) + '\..\Results\_borrar_testing_soc2{}_{}.csv'.format(opt_method.__name__, data_type), index=False)
    return final_results, betas_results

if __name__ == "__main__":
    results, betas = main()
