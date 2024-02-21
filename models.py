# -*- coding: utf-8 -*-
#%%
"""
Spyder Editor

This is a temporary script file.
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "10"
import cvxpy as cp

import numpy as np
import matplotlib.pyplot as plt

from time import time, process_time

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, LassoLars 
from scipy.optimize import minimize

# from model_solver import GUROBI_MODEL


# Random seed
np.random.seed(123)

# Lasso from the scikit-learn package
def SCIKIT_LASSO(X,y, tau, solver=cp.MOSEK, solver_params={}, solver_verbose=False, lasso_type='Lasso', tol=1e-6):
    """
        lasso_type: str. 
            'Lasso': Lasso from the scikit-learn package
            'LassoLars': LassoLars from the scikit-learn package
    """
    N,M = X.shape
    # =============================================================================
    #                                  SCIKIT LASSO
    # =============================================================================
    print("""
    --------------------------------------------------
                        SCIKIT LASSO Model  
    --------------------------------------------------
    """)
    t0 = time()
    t0p = process_time()

    # =============================================================================
    #                             Model
    # =============================================================================
    
    # alpha = 2*tau/(2*n), because f.o.: 1/(2n)norm2 + alpha*norm1
    lasso = Lasso if lasso_type == 'Lasso' else LassoLars
    clf = lasso(
        alpha=tau/N, 
        fit_intercept=False,
        max_iter=1/tol,
        tol=tol
        ) 
    clf.fit(X, y)
    beta = clf.coef_

    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    z = beta # to fit the same format as the other models

    lasso_value = np.linalg.norm(y - X @ beta)**2 + 2*tau * np.linalg.norm(beta, 1)

    return beta, z, lasso_value, (t1-t0)/60, (t1p-t0p)/60, {}
 
# Elastic Net from the scikit-learn package
def SCIKIT_ElasticNet(X,y, tau, theta=0, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    N,M = X.shape
    # =============================================================================
    #                             SCIKIT Elastic Net
    # =============================================================================
    print("""
    --------------------------------------------------
                SCIKIT Elastic Net Model  
    --------------------------------------------------
    """)
    t0 = time()
    t0p = process_time()

    # =============================================================================
    #                             Model
    # =============================================================================
    
    # alpha = 2*tau/(2*n), because f.o.: 1/(2n)norm2 + alpha*norm1
    clf = ElasticNet(
        alpha=(2*tau + theta)/(2*N), 
        l1_ratio=2*tau/(2*tau + theta),
        fit_intercept=False,
        max_iter=1e6,
        tol=1e-6
        ) 
    clf.fit(X, y)
    beta = clf.coef_

    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    z = beta # to fit the same format as the other models

    elastic_net_value = np.linalg.norm(y - X @ beta)**2 + (2 * tau ) * np.linalg.norm(beta, 1) + (0.5 * theta) * np.linalg.norm(beta)**2

    return beta, z, elastic_net_value, (t1-t0)/60, (t1p-t0p)/60, {}

# Lasso as its relaxed form
def L_LASSO(X,y, tau_tilda, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    N,M = X.shape
    # =============================================================================
    #                                   L. LASSO
    # =============================================================================

    print("""
    --------------------------------------------------
                     L. LASSO Model
    --------------------------------------------------
    """)

    t0 = time()
    t0p = process_time()
    
    # =============================================================================
    #                             Model Variables
    # =============================================================================
    
    # 1. Continuous unbounded
    beta = cp.Variable(M, name="beta", nonneg=False, boolean=False, integer=False)
    
    # =============================================================================
    #                             Objective Function
    # =============================================================================

    # Identity square matrix of nxn
    I = np.eye(N)

    # 2. Objective function
    lasso = cp.Problem(
        cp.Minimize(
            cp.quad_form(y - X @ beta, I) + 2*tau_tilda * cp.norm(beta, 1) # (y - X beta)' I (y - X beta) 
        )
    ) # 2*tau is because it represents tau + kappa against error^2
    
    lasso.solve(
        verbose=solver_verbose, 
        solver=solver, 
        **solver_params
        # warm_start=True,
        # Threads=10
        )
    
    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    z = beta # to fit the same format as the other models

    return beta.value, z.value, lasso.value, (t1-t0)/60, (t1p-t0p)/60, {}

# Lasso as its relaxed form, but solved with scipy [NOT STABLE SOLUTION. MAYBE DELETE IT]
def L_LASSO_scipy(X,y, tau_tilda, solver='L-BFGS-B', solver_params={}, tol=1e-6):
    N,M = X.shape
    # =============================================================================
    #                                   L. LASSO
    # =============================================================================

    print("""
    --------------------------------------------------
                     L. LASSO Model (scipy)
    --------------------------------------------------
    """)

    t0 = time()
    t0p = process_time()
    
    # =============================================================================
    #                             Model function
    # =============================================================================
    
    def lasso_obj(beta):
        return np.linalg.norm(y - X @ beta)**2 + 2*tau_tilda * np.linalg.norm(beta, 1)
    
    # =============================================================================
    #                             Model
    # =============================================================================

    # 2. Objective function
    res = minimize(lasso_obj, np.zeros(M), method=solver, options=solver_params, tol=tol)
    beta = res.x

    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    z = beta # to fit the same format as the other models

    lasso_value = np.linalg.norm(y - X @ beta)**2 + 2*tau_tilda * np.linalg.norm(beta, 1)

    return beta, z, lasso_value, (t1-t0)/60, (t1p-t0p)/60, {}
    

# The original MIQP that we proposed (same of Bertsimas)
def MIQP(X,y, tau_tilda, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    N,M = X.shape
     # =============================================================================
    #                                   MIQP
    # =============================================================================
    
    # Acotar el MIP
    # m.setParam(GRB.Param.MIPGapAbs, 1e-1000)  # Gurobi should stop once |z_upper - z_lower | < 1e-1000
    # m.setParam(GRB.Param.Threads, 10)           # Gurobi cores assigned to the problem
    # m.setParam(GRB.Param.QCPDual, 1)            # Gurobi should calculate the dual variables
    # m.setParam(GRB.Param.BarQCPConvTol, 1e-6)   # Gurobi should stop once the QCP barrier converges (dual-primal gap < 1e-6)
    # m.setParam( 'OutputFlag', False )         # Gurobi should not print anything

    t0 = time()
    t0p = process_time()
    
    # =============================================================================
    #                             Model Variables
    # =============================================================================
    
    # 1. Continuous unbounded
    beta = cp.Variable(M, name="beta", nonneg=False, boolean=False, integer=False)

    # 2. Binary
    z = cp.Variable(M, name="z", boolean=True)

    # =============================================================================
    #                             Model Constraints
    # =============================================================================

    big_M = 1e6

    # 3. Cone 2_i: Binary form of the beta_i <= z_i * u_i constraint 
    soc_2 = [
        -beta[i] <= big_M * z[i] for i in range(M)
    ] + \
    [
        beta[i]  <= big_M * z[i] for i in range(M)
    ]

    
    # =============================================================================
    #                             Objective Function
    # =============================================================================
    
    # # 1. Penalization of the coefficients
    # tau = error_quad_OLS/M
    # kappa = tau

    # tau_tilda = np.sqrt(error_quad_OLS)/(M**m_exp_rate)
    # kappa_tilda = tau_tilda

    # Identity square matrix of nxn
    I = np.eye(N)

    # 2. Objective function
    miqp = cp.Problem(
        cp.Minimize(
            cp.quad_form(y - X @ beta, I) + tau_tilda * (np.ones(M) @ z) 
        ),
        soc_2
    )
    
    # t0 = time()
    # t0p = process_time()
    
    miqp.solve(
        verbose=solver_verbose, 
        solver=solver, 
        **solver_params
        # warm_start=True,
        # Threads=10
        )
    
    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    print('z value', z.value)

    return beta.value, z.value, miqp.value, (t1-t0)/60, (t1p-t0p)/60, {}


def MIP_R(X,y,tau_tilda,kappa_tilda, eps_sqrt=0, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    N,M = X.shape
     # =============================================================================
    #                                   SOCP
    # =============================================================================
    
    # Acotar el MIP
    # m.setParam(GRB.Param.MIPGapAbs, 1e-1000)  # Gurobi should stop once |z_upper - z_lower | < 1e-1000
    # m.setParam(GRB.Param.Threads, 10)           # Gurobi cores assigned to the problem
    # m.setParam(GRB.Param.QCPDual, 1)            # Gurobi should calculate the dual variables
    # m.setParam(GRB.Param.BarQCPConvTol, 1e-6)   # Gurobi should stop once the QCP barrier converges (dual-primal gap < 1e-6)
    # m.setParam( 'OutputFlag', False )         # Gurobi should not print anything

    t0 = time()
    t0p = process_time()
    
    # =============================================================================
    #                             Model Variables
    # =============================================================================
    
    # 1. Continuous unbounded
    beta = cp.Variable(M, name="beta", nonneg=False, boolean=False, integer=False)
    xi   = cp.Variable(1, name="xi",   nonneg=False, boolean=False, integer=False)


    # 2. Continuous positive 
    u = cp.Variable(M, name="u", nonneg=True, boolean=False, integer=False)
    z = cp.Variable(M, name="z", nonneg=True, boolean=False, integer=False)

    # 3. Aux vector
    aux_vector = cp.Variable((M, 2 + (eps_sqrt!=0)), name="aux_vector", nonneg=False, boolean=False, integer=False)
    aux = cp.Variable(N, name="aux", nonneg=False, boolean=False, integer=False)
    
    # =============================================================================
    #                             Model Constraints
    # =============================================================================
    
    # 1. Auxiliar constraint for the quadratic multiplication of MVars on left side

    # 2. Cone 1: Linnearization of the residuals norm
    # soc_1 = [
    #     cp.SOC(
    #         xi, 
    #         y - X @ beta
    #         )
    # ]
    soc_1 = [ y - X @ beta == aux ]
    soc_1 += [aux @ aux <= xi ** 2]

    # 2. Auxiliar vector
    aux_constr = [
        aux_vector[i,0] == (u[i] - z[i])/2 for i in range(M)
    ]
    aux_constr += [
        aux_vector[i,1] == beta[i] for i in range(M)
    ]
    if eps_sqrt != 0:
        aux_constr += [
            aux_vector[i,2] == eps_sqrt for i in range(M)
        ]

    # 3. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
    soc_2 = [
        cp.SOC(
            ( u[i] + z[i] )/2,
            aux_vector[i,:]
            ) for i in range(M)
    ]
    
    # =============================================================================
    #                             Objective Function
    # =============================================================================
    
    # # 1. Penalization of the coefficients
    # tau = error_quad_OLS/M
    # kappa = tau

    # tau_tilda = np.sqrt(error_quad_OLS)/(M**m_exp_rate)
    # kappa_tilda = tau_tilda

    # 2. Objective function
    socp = cp.Problem(
        cp.Minimize(
            xi + tau_tilda * (np.ones(M) @ z) + kappa_tilda * (np.ones(M) @ u)
        ),
        soc_1 + aux_constr + soc_2
    )
    
    # t0 = time()
    # t0p = process_time()
    
    socp.solve(
        verbose=solver_verbose, 
        solver=solver, 
        **solver_params
        # warm_start=True,
        # Threads=10
        )
    
    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    return beta.value, xi.value, u.value, z.value, socp.value, (t1-t0)/60, (t1p-t0p)/60, {}

# The original SOCP that we proposed 
def SOCP(X,y,tau, kappa, theta=0, eps_sqrt=0, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    print("""
    --------------------------------------------------
                        SOCP Model
    --------------------------------------------------
    """)

    N,M = X.shape
     # =============================================================================
    #                                   SOCP
    # =============================================================================
    
    t0 = time()
    t0p = process_time()
    
    # =============================================================================
    #                             Model Variables
    # =============================================================================
    
    # 1. Continuous unbounded
    beta = cp.Variable(M, name="beta", nonneg=False, boolean=False, integer=False)
    xi   = cp.Variable(1, name="xi",   nonneg=False, boolean=False, integer=False)
    xi_2  = cp.Variable(1, name="xi_2",   nonneg=False, boolean=False, integer=False)


    # 2. Continuous positive 
    u = cp.Variable(M, name="u", nonneg=True, boolean=False, integer=False)
    z = cp.Variable(M, name="z", nonneg=True, boolean=False, integer=False)

    # 3. Aux vector
    aux_vector = cp.Variable((M, 2 + (eps_sqrt!=0)), name="aux_vector", nonneg=False, boolean=False, integer=False)
    
    # =============================================================================
    #                             Model Constraints
    # =============================================================================
    
    # 1. Auxiliar constraint for the quadratic multiplication of MVars on left side

    # 2. Cone 1: Linnearization of the residuals norm
    soc_1 = [
        cp.SOC(
            xi, 
            y - X @ beta
            )
    ]

    # 2. Auxiliar vector
    aux_constr = [
        aux_vector[i,0] == (u[i] - z[i]) for i in range(M)
    ]
    aux_constr += [
        aux_vector[i,1] == 2*beta[i] for i in range(M)
    ]
    if eps_sqrt != 0:
        aux_constr += [
            aux_vector[i,2] == eps_sqrt for i in range(M)
        ]

    # 3. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
    soc_2 = [
        cp.SOC(
            ( u[i] + z[i] ),
            aux_vector[i,:]
            ) for i in range(M)
    ]
    
    # 4. Cone 3: Elastic net constraint (if theta > 0)
    soc_3 = [
        cp.SOC(
            xi_2,
            beta
            )
    ]
    # =============================================================================
    #                             Objective Function
    # =============================================================================
    
    # 1. Objective function
    socp = cp.Problem(
        cp.Minimize(
            xi**2 + tau * (np.ones(M) @ z) + kappa * (np.ones(M) @ u) + 0.5 * theta * xi_2**2
        ),
        soc_1 + aux_constr + soc_2 + soc_3
    )

    
    # t0 = time()
    # t0p = process_time()
    
    socp.solve(
        verbose=solver_verbose, 
        solver=solver, 
        **solver_params
        # warm_start=True,
        # Threads=10
        )
    
    t1 = time()
    t1p = process_time()
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")

    return beta.value, xi.value, u.value, z.value, xi_2.value, socp.value, (t1-t0)/60, (t1p-t0p)/60, {}


# Dummy auxiliar class
class Dummy:
    def __init__(self) -> None:
        self.value = None

# It should be the same master problem for all 3 cases
def masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    n,m = X.shape
    e_m = np.ones(m) 
    # e_k = np.ones(k+v-1)
    e_k = np.ones((v0-1)+k+(v-1)*k_v)
    # =============================================================================
    #                                Dual Model
    # =============================================================================

    # =============================================================================
    #                             Model Variables
    # =============================================================================
    
    # 1. Con1 dual vars
    psi_k = cp.Variable(n, name="psi_k")
    mu_k = cp.Variable(1, name="mu_k")

    # 2. Cone 2 dual vars
    alpha_gamma_k = cp.Variable((m,2), name="alpha_gamma_k")
    delta_k = cp.Variable(m, name="delta_k")

    # 3. Sum(pi) = 1 constraint
    if sum_1_comb:
        sigma = cp.Variable(1, name="sigma")
    else:
        sigma = 0
    # =============================================================================
    #                             Model Constraints
    # =============================================================================

    # 1. Cone 1: Linnearization of the residuals norm DUAL
    soc_1 = [
        cp.SOC(
            mu_k, 
            psi_k
            )
    ]

    # 2. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
    soc_2 = [
        cp.SOC(
            delta_k[i], 
            alpha_gamma_k[i,:]	
            ) for i in range(m)
    ]
    

    # 3. Lagrangian bounding constraints
    if pos_linear_comb:
        const = [	
            (tau_tilda * e_m + alpha_gamma_k[:,0] - delta_k) @ z_k \
            + (kappa_tilda * e_m - alpha_gamma_k[:,0] - delta_k) @ u_k \
            + (psi_k.T @ X - 2*alpha_gamma_k[:,1].T) @ beta_k \
            + sigma*e_k \
                >= 0
        ]
    else:
        const = [	
            (tau_tilda * e_m.T - alpha_gamma_k[:,0].T - delta_k.T) @ z_k \
            + (kappa_tilda * e_m.T + alpha_gamma_k[:,0].T - delta_k.T) @ u_k \
            + (psi_k.T @ X - 2*alpha_gamma_k[:,1].T) @ beta_k \
            + sigma*e_k \
                == 0
        ] 


    # # 4. [TEST] Lagragian -inf restriction: abs(psi_k.T @ X) <= trau_tilda + kappa_tilda (linnearize this constraint)
    # if psi_X_contraint:
    #     const += [
    #         psi_k.T @ X <= tau_tilda + kappa_tilda,
    #         - psi_k.T @ X <= tau_tilda + kappa_tilda
    #     ]

    # =============================================================================
    #                             Objective Function
    # =============================================================================

    # 1. Objective function
    obj = cp.Maximize(
        - mu_k **2/4 - psi_k @ y - sigma 
    )

    # =============================================================================
    #                             Model
    # =============================================================================

    # 1. Create model
    dual = cp.Problem(obj, soc_1 + soc_2 + const)

    # times
    t0 = time()
    t0p = process_time()

    # 2. Solve dual
    dual.solve(
        solver=solver, 
        verbose=solver_verbose,
        **solver_params
        # Threads=10,
        # BarConvTol=gp_tol,
        # BarQCPConvTol=gp_tol,
        )

    # times
    t1 = time()
    t1p = process_time()

    print("-"*50)
    print("The optimal DUAL value is", -dual.value)
    print("Time elapsed in DUAL of iteration ", k)
    print((t1-t0)/60, " mins (normal)")
    print((t1p-t0p)/60, " mins (process)")
    print("-"*50)


    # 3. Get solution
    psi_k = psi_k.value
    mu_k = mu_k.value
    alpha_gamma_k = alpha_gamma_k.value
    delta_k = delta_k.value

    # print("dual status:", dual.status)
    # print("dual value:", dual.value)

    # # Print dual variables
    # print("psi_k", psi_k)
    # print("mu_k", mu_k)
    # print("alpha_gamma_k", alpha_gamma_k)
    # print("delta_k", delta_k)

    return psi_k, mu_k, alpha_gamma_k, delta_k, (t1-t0)/60, (t1p-t0p)/60

def convex_linnear_regression(X,y, pos_linear_comb=False , solver=cp.MOSEK, solver_params={}, solver_verbose=False):
    # =============================================================================
    #                               Linnear Model
    # =============================================================================

    # =============================================================================
    #                             Model Variables
    # =============================================================================

    # 1. Beta variables
    beta = cp.Variable(X.shape[1], name="beta", nonneg=pos_linear_comb)

    # 2. Residuals variables
    e = cp.Variable(X.shape[0], name="e")

    # =============================================================================
    #                             Model Constraints
    # =============================================================================

    # 1. Betas must sum 1 (convex combination)
    sum = [cp.sum(beta) == 1]

    # 2. Residuals as error
    error = [e == X @ beta - y]

    # =============================================================================
    #                             Objective Function
    # =============================================================================
    
    I = np.eye(X.shape[0])
    # 1. Objective function
    obj = cp.Minimize(cp.quad_form(e, I))

    # =============================================================================
    #                             Model
    # =============================================================================

    # 1. Create model
    linnear = cp.Problem(obj, sum + error)

    # 2. Solve linnear
    linnear.solve(
        solver=solver, 
        verbose=False,
        # Threads=10
        )

    # 3. Get solution
    beta = beta.value
    e = e.value

    # print("linnear status:", linnear.status)
    # print("linnear value:", linnear.value)

    # # Print linnear variables
    # print("beta", beta)
    # print("e", e)

    return beta, e



# Original version with the psi'X >= tau + kappa condition
def CG_SOC1_upgrade(X,y,tau_tilda,kappa_tilda, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    # np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        rand_beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        rand_beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        rand_beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        rand_beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, rand_beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

        # new (incluir sol. OLS)

        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # reshape beta_OLs to (m,1)
        # beta_k = np.concatenate((beta_OLS.reshape(-1,1), rand_beta_k), axis=1)
        # v0 +=1

        # # new
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # beta_k[:,1] = -1 # second column is -1
        # v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    z_k[beta_k != 0] = 1*np.sqrt(kappa_tilda/tau_tilda)
    u_k[beta_k != 0] = 1*np.sqrt(tau_tilda/kappa_tilda)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # if save_conv_info:
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ beta_k @ pi_k
                )
        ]

        # 2. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
        aux_matrix = []
        for i in range(m):
            aux_matrix.append(np.array([( u_k[i,:] - z_k[i,:] ), 2*beta_k[i,:], eps_soc2_sqrt_vector]))
        # print("aux matrix", aux_matrix)
        soc_2 = [
            cp.SOC(
                ( u_k[i,:] + z_k[i,:] ) @ pi_k, # epsilon soc2 is the anti-vertex gap (appears to be useless)
                aux_matrix[i] @ pi_k
                ) for i in range(m)
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # 4. u_k and z_k must be positive
        u_z_pos = [
            u_k @ pi_k >= 0,
            z_k @ pi_k >= 0
        ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + tau_tilda * cp.sum( z_k @ pi_k ) + kappa_tilda * cp.sum( u_k @ pi_k )
            ),
            soc_1 + soc_2 + wei_sum + u_z_pos
        )

        # print("Master problem values")
        # print("tau_tilda:", tau_tilda)
        # print("kappa_tilda:", kappa_tilda)
        # print("beta_k:", beta_k)
        # print("X", X )
        # print("X @ beta_k:", X @ beta_k)
        # print("y:", y)
        # print("z_k:", z_k)
        # print("u_k:", u_k)
        

        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            # BarConvTol=gp_tol,
            # BarQCPConvTol=gp_tol,
            # Threads=10
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        # print('beta_k',beta_k)
        # print('beta_k.shape', beta_k.shape)
        # print('pi_k_sol', pi_k_sol)
        # print('pi_k_sol.shape', pi_k_sol.shape)
        beta_k_sol = beta_k @ pi_k_sol
        z_k_sol = z_k @ pi_k_sol
        u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")
                
                psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb,  solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                solved_duals.append(k)
        else:
            psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            if k in solved_duals:
                dual_times.append(dual_time)
                dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )

        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            # if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol: # OLD
            if np.max(np.abs(lambda_k_diff)) < cg_lambda_tol:

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                # print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                # print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================
        # print((m, v))
        # Check condition of |psi_k_sol.T @ X| > tau_tilda + kappa_tilda (-inf inmediately) 
        # It's a best bound than |psi_k_sol.T @ X| > 0. Only possible because of b_i^2 <= z_i*u_i
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > tau_tilda + kappa_tilda).any() if not dummy_condition else False
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa_tilda*tau_tilda)).any() if not dummy_condition else False

        # 
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa_tilda*tau_tilda) - cg_lambda_tol).any() if not dummy_condition else False
        
        if not lagrange_boundedness_conditon:
            # print('condition min diff:')
            # print(np.max(np.abs(psi_k_sol.T @ X)) - 2*np.sqrt(kappa_tilda*tau_tilda))


            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            # xi = cp.Variable(1, name="xi", nonneg=True)

            # 2. Continuous positive 
            z = cp.Variable(m, name="z", nonneg=True)
            u = cp.Variable(m, name="u", nonneg=True)

            # 3. Aux vector
            aux_vector = cp.Variable((m, 2 + (eps_soc2_sqrt_L!=0)), name="aux_vector", nonneg=False, boolean=False, integer=False)

            # =============================================================================
            #                             Objective Function
            # =============================================================================

            # 1. Vector of conic constraints
            # cone1_vec = np.array([xi, y - X @ beta])
            # cone2_vec = np.array([z, beta])

            # 2.1 Auxiliar vector
            aux_constr = [
                aux_vector[i,0] == (u[i] - z[i]) for i in range(m)
            ]
            aux_constr += [
                aux_vector[i,1] == 2*beta[i] for i in range(m)
            ]
            if eps_soc2_sqrt_L != 0:
                aux_constr += [
                    aux_vector[i,2] == eps_soc2_sqrt_L for i in range(m)
                ]


            # 2.2 Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
            soc_2_L = [
                cp.SOC(
                    ( u[i] + z[i] ), # epsilon soc2 is the anti-vertex gap
                    aux_vector[i,:]
                    ) for i in range(m)
            ]

            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    # + xi**2 - mu_k_sol*xi  # irrelevant for this case
                    + (- mu_k_sol/2) * mu_k_sol/2 \
                    + psi_k_sol.T @ X @ beta \
                    + tau_tilda * cp.sum(z) \
                    + kappa_tilda * cp.sum(u) \
                    - psi_k_sol.T @ y # Constant (may be removed)
                ),
                aux_constr + soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    **solver_params
                    # Threads=10,
                    # BarConvTol=gp_tol,
                    # BarQCPConvTol=gp_tol,
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model [Error while solving Lagrangian Model]")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            # print lagrangian times
            print('Time elapsed in Lagrange of iteration  6')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)

        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            psi_k_sol_X = psi_k_sol.T @ X
            if psi_k_sol_X.shape[0] == 1:	# If is shape (1,50)
                v_constraint        = psi_k_sol_X[0]
            else:                           # Shape (50,)
                v_constraint        = psi_k_sol_X
            
            # Option 1: n indices
            v_indices           = np.argpartition(np.abs(v_constraint), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)
            z_sol       = np.zeros((m, v), dtype=float)
            u_sol       = np.zeros((m, v), dtype=float)

            # positive beta
            v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
            s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices_pos, s_pos] = 1
            z_sol[v_indices_pos, s_pos] = 1*np.sqrt(kappa_tilda/tau_tilda)
            u_sol[v_indices_pos, s_pos] = 1*np.sqrt(tau_tilda/kappa_tilda)

            # negative beta
            v_indices_neg = v_indices[v_constraint[v_indices] > 0]
            s_neg = range(len(v_indices_pos),v)#range(len(v_indices_neg))
            beta_sol[v_indices_neg, s_neg] = -1
            z_sol[v_indices_neg, s_neg] = 1*np.sqrt(kappa_tilda/tau_tilda)
            u_sol[v_indices_neg, s_neg] = 1*np.sqrt(tau_tilda/kappa_tilda)

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            z_sol      = z.value
            u_sol      = u.value
            xi_sol     = mu_k_sol/2


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            # if np.sqrt(residuals @ residuals) < cg_residuals_tol: # OLD
            if np.max(np.abs(residuals)) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # add the following to the return final dict
                # master_values = []
                # lagrangian_values = []
                # dual_values = []

                # # solver times
                # master_times = []
                # lagrangian_times = []
                # dual_times = []

                # # solver p_times
                # master_p_times = []
                # lagrangian_p_times = []
                # dual_p_times = []
                # solved_duals = []
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0") # , np.sqrt(residuals @ residuals)

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)
            z_k = np.append(z_k, z_sol.reshape(m,1), axis=1)
            u_k = np.append(u_k, u_sol.reshape(m,1), axis=1)
        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)
            z_k = np.append(z_k, z_sol, axis=1)
            u_k = np.append(u_k, u_sol, axis=1)

        k += 1

def CG_SOC1_ElasticNet(X,y,tau,kappa, theta=0, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    # np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        rand_beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        rand_beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        rand_beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        rand_beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, rand_beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

        # new (incluir sol. OLS)

        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # reshape beta_OLs to (m,1)
        # beta_k = np.concatenate((beta_OLS.reshape(-1,1), rand_beta_k), axis=1)
        # v0 +=1

        # # new
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # beta_k[:,1] = -1 # second column is -1
        # v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    z_k[beta_k != 0] = 1*np.sqrt(kappa/tau)
    u_k[beta_k != 0] = 1*np.sqrt(tau/kappa)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # if save_conv_info:
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        xi_k_2 = cp.Variable(1, name="xi_k_2")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ beta_k @ pi_k
                )
        ]

        # 2. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
        aux_matrix = []
        for i in range(m):
            aux_matrix.append(np.array([( u_k[i,:] - z_k[i,:] ), 2*beta_k[i,:], eps_soc2_sqrt_vector]))
        # print("aux matrix", aux_matrix)
        soc_2 = [
            cp.SOC(
                ( u_k[i,:] + z_k[i,:] ) @ pi_k, # epsilon soc2 is the anti-vertex gap (appears to be useless)
                aux_matrix[i] @ pi_k
                ) for i in range(m)
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # 4. u_k and z_k must be positive
        u_z_pos = [
            u_k @ pi_k >= 0,
            z_k @ pi_k >= 0
        ]

        # 5. Elastic net constraint
        soc_3 = [
            cp.SOC(
                xi_k_2, 
                beta_k @ pi_k   
            )
        ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + tau * cp.sum( z_k @ pi_k ) + kappa * cp.sum( u_k @ pi_k ) + 0.5 * theta * xi_k_2**2
            ),
            soc_1 + soc_2 + wei_sum + u_z_pos + soc_3
        )

        # print("Master problem values")
        # print("tau:", tau)
        # print("kappa:", kappa)
        # print("beta_k:", beta_k)
        # print("X", X )
        # print("X @ beta_k:", X @ beta_k)
        # print("y:", y)
        # print("z_k:", z_k)
        # print("u_k:", u_k)
        

        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            # BarConvTol=gp_tol,
            # BarQCPConvTol=gp_tol,
            # Threads=10
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        # print('beta_k',beta_k)
        # print('beta_k.shape', beta_k.shape)
        # print('pi_k_sol', pi_k_sol)
        # print('pi_k_sol.shape', pi_k_sol.shape)
        beta_k_sol = beta_k @ pi_k_sol
        z_k_sol = z_k @ pi_k_sol
        u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value
        xi_k_2_sol = xi_k_2.value


        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)
            soc_3_k_sol = np.array([soc_3[x].dual_value for x in range(len(soc_3))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]

                phi_k_sol = soc_3_k_sol[0][1]
                mu_2_k_sol = soc_3_k_sol[0][0][0]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")
                
                psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb,  solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                solved_duals.append(k)
        else:
            psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            if k in solved_duals:
                dual_times.append(dual_time)
                dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol)
        lambda_k = np.append(lambda_k, phi_k_sol)
        lambda_k = np.append(lambda_k, mu_2_k_sol)

        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            # if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol: # OLD
            if np.max(np.abs(lambda_k_diff)) < cg_lambda_tol:

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, xi_k_2_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                # print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                # print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================
        # print((m, v))
        # Check condition of |psi_k_sol.T @ X| > tau + kappa (-inf inmediately) 
        # It's a best bound than |psi_k_sol.T @ X| > 0. Only possible because of b_i^2 <= z_i*u_i
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > tau + kappa).any() if not dummy_condition else False
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa*tau)).any() if not dummy_condition else False
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X - phi_k_sol) - 2*np.sqrt(kappa*tau) > -cg_lambda_tol).any() if not dummy_condition else False
        
        if not lagrange_boundedness_conditon:
            # print('condition min diff:')
            # print(np.max(np.abs(psi_k_sol.T @ X - phi_k_sol) - 2*np.sqrt(kappa*tau)))

            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            # xi = cp.Variable(1, name="xi", nonneg=True)¨
            xi_2 = cp.Variable(1, name="xi_2", nonneg=True)

            # 2. Continuous positive 
            z = cp.Variable(m, name="z", nonneg=True)
            u = cp.Variable(m, name="u", nonneg=True)

            # 3. Aux vector
            aux_vector = cp.Variable((m, 2 + (eps_soc2_sqrt_L!=0)), name="aux_vector", nonneg=False, boolean=False, integer=False)

            # =============================================================================
            #                             Objective Function
            # =============================================================================

            # 1. Vector of conic constraints
            # cone1_vec = np.array([xi, y - X @ beta])
            # cone2_vec = np.array([z, beta])

            # 2.1 Auxiliar vector
            aux_constr = [
                aux_vector[i,0] == (u[i] - z[i]) for i in range(m)
            ]
            aux_constr += [
                aux_vector[i,1] == 2*beta[i] for i in range(m)
            ]
            if eps_soc2_sqrt_L != 0:
                aux_constr += [
                    aux_vector[i,2] == eps_soc2_sqrt_L for i in range(m)
                ]


            # 2.2 Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
            soc_2_L = [
                cp.SOC(
                    ( u[i] + z[i] ), # epsilon soc2 is the anti-vertex gap
                    aux_vector[i,:]
                    ) for i in range(m)
            ]

            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    # + (xi - mu_k_sol)*xi  # irrelevant for this case (direct solution down below with xi* = mu_k_sol / 2)
                    + (- mu_k_sol/2) * mu_k_sol/2 \
                    + psi_k_sol.T @ X @ beta \
                    + tau * cp.sum(z) \
                    + kappa * cp.sum(u) \
                     # Constant down below (may be removed)
                    - psi_k_sol.T @ y \
                    # + (0.5 * theta * xi_2 - mu_2_k_sol) * xi_2 # irrelevant for this case (direct solution down below with xi_2* = mu_2_k_sol / theta)
                    + (- mu_2_k_sol/2) * mu_2_k_sol/theta
                ),
                aux_constr + soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    **solver_params
                    # Threads=10,
                    # BarConvTol=gp_tol,
                    # BarQCPConvTol=gp_tol,
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            # print lagrangian times
            print('Time elapsed in Lagrange of iteration  6')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)

        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            psi_k_sol_X = psi_k_sol.T @ X
            if psi_k_sol_X.shape[0] == 1:	# If is shape (1,50)
                v_constraint        = psi_k_sol_X[0]
            else:                           # Shape (50,)
                v_constraint        = psi_k_sol_X
            
            # Option 1: n indices
            v_indices           = np.argpartition(np.abs(v_constraint), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)
            z_sol       = np.zeros((m, v), dtype=float)
            u_sol       = np.zeros((m, v), dtype=float)

            # positive beta
            v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
            s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices_pos, s_pos] = 1
            z_sol[v_indices_pos, s_pos] = 1*np.sqrt(kappa/tau)
            u_sol[v_indices_pos, s_pos] = 1*np.sqrt(tau/kappa)

            # negative beta
            v_indices_neg = v_indices[v_constraint[v_indices] > 0]
            s_neg = range(len(v_indices_pos),v)#range(len(v_indices_neg))
            beta_sol[v_indices_neg, s_neg] = -1
            z_sol[v_indices_neg, s_neg] = 1*np.sqrt(kappa/tau)
            u_sol[v_indices_neg, s_neg] = 1*np.sqrt(tau/kappa)

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            z_sol      = z.value
            u_sol      = u.value
            xi_sol     = mu_k_sol/2
            xi_2_sol   = mu_2_k_sol/theta


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            # if np.sqrt(residuals @ residuals) < cg_residuals_tol: # OLD
            if np.max(np.abs(residuals)) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, xi_k_2_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0") # , np.sqrt(residuals @ residuals)

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)
            z_k = np.append(z_k, z_sol.reshape(m,1), axis=1)
            u_k = np.append(u_k, u_sol.reshape(m,1), axis=1)
        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)
            z_k = np.append(z_k, z_sol, axis=1)
            u_k = np.append(u_k, u_sol, axis=1)

        k += 1


# LASSO CG
def CG_LASSO_SOC1(X,y,tau,kappa, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    # np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        rand_beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        rand_beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        rand_beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        rand_beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, rand_beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    # z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # z_k[beta_k != 0] = 1*np.sqrt(kappa/tau)
    # u_k[beta_k != 0] = 1*np.sqrt(tau/kappa)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        print("-"*50)
        print("Master Problem")
        print("-"*50)

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        eta_k = cp.Variable(1, name="eta_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ beta_k @ pi_k
                )
        ]

        # 2. Cone 2: First order cone of beta: ||beta||_1 <= eta
        soc_2 = [
            cp.norm1(beta_k @ pi_k) <= eta_k
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function 
            
        # ( lamda = 2*sqrt(kappa*tau) ) => ( tau=kappa => lambda=2*tau )
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + 2 * np.sqrt(tau*kappa) * eta_k
            ),
            soc_1 + soc_2 + wei_sum
        )


        # print("Master problem values")
        # print("tau:", tau)
        # print("kappa:", kappa)
        # print("beta_k:", beta_k)
        # print("X", X )
        # print("X @ beta_k:", X @ beta_k)
        # print("y:", y)
        # print("z_k:", z_k)
        # print("u_k:", u_k)
        

        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        # print('beta_k',beta_k)
        # print('beta_k.shape', beta_k.shape)
        # print('pi_k_sol', pi_k_sol)
        # print('pi_k_sol.shape', pi_k_sol.shape)
        beta_k_sol = beta_k @ pi_k_sol
        # z_k_sol = z_k @ pi_k_sol
        # u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value
        eta_k_sol = eta_k.value
        

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")

                return None
                
                # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb,  solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                # solved_duals.append(k)
        else:
            return None
            # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            # solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            # if k in solved_duals:
            #     dual_times.append(dual_time)
            #     dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )

        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            # if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol: # OLD
            if np.max(np.abs(lambda_k_diff)) < cg_lambda_tol: # infinity norm < tol

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, eta_k_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                # print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                # print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================
        # print((m, v))
        # Check condition of |psi_k_sol.T @ X| > tau + kappa (-inf inmediately) 
        # It's a best bound than |psi_k_sol.T @ X| > 0. Only possible because of b_i^2 <= z_i*u_i
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > tau + kappa).any() if not dummy_condition else False
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa*tau)).any() if not dummy_condition else False
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau) > -cg_lambda_tol).any() if not dummy_condition else False
        
        if not lagrange_boundedness_conditon:
            # print('condition min diff:')
            # print(np.max(np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau)))

            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")
            print("-"*50)

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            eta = cp.Variable(1, name="eta", nonneg=True)

            # =============================================================================
            #                             Objective Function
            # =============================================================================


            # 2.2 Cone 2: ||beta||_1 <= eta
            soc_2_L = [
                cp.norm1(beta) <= eta
            ]

            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    # + xi**2 - mu_k_sol*xi  # irrelevant for this case
                    + (- mu_k_sol/2) * mu_k_sol/2 \
                    + psi_k_sol.T @ X @ beta \
                    +  2 * np.sqrt(tau*kappa) * eta \
                    - psi_k_sol.T @ y # Constant (may be removed)
                ),
                soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    **solver_params
                    # Threads=10,
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            # print lagrangian times
            print('Time elapsed in Lagrange of iteration  6')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)

        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            psi_k_sol_X = psi_k_sol.T @ X
            if psi_k_sol_X.shape[0] == 1:	# If is shape (1,50)
                v_constraint        = psi_k_sol_X[0]
            else:                           # Shape (50,)
                v_constraint        = psi_k_sol_X
            
            # Option 1: n indices
            v_indices           = np.argpartition(np.abs(v_constraint), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)

            # positive beta
            v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
            s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices_pos, s_pos] = 1

            # negative beta
            v_indices_neg = v_indices[v_constraint[v_indices] > 0]
            s_neg = range(len(v_indices_pos),v)#range(len(v_indices_neg))
            beta_sol[v_indices_neg, s_neg] = -1

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            xi_sol     = mu_k_sol/2


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            # if np.sqrt(residuals @ residuals) < cg_residuals_tol: # OLD
            if np.max(np.abs(residuals)) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	


                return beta_final_sol, xi_k_sol, eta_k_sol, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0") # , np.sqrt(residuals @ residuals)

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)

        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)

        k += 1

# LASSO CG with f.o. l1 instead of constraint.
def CG_LASSO_SOC1_v2(X,y,tau,kappa, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    # np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        rand_beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        rand_beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        rand_beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        rand_beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, rand_beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    # z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # z_k[beta_k != 0] = 1*np.sqrt(kappa/tau)
    # u_k[beta_k != 0] = 1*np.sqrt(tau/kappa)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        print("-"*50)
        print("Master Problem")
        print("-"*50)

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        eta_k = cp.Variable(1, name="eta_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ beta_k @ pi_k
                )
        ]

        # # 2. Cone 2: First order cone of beta: ||beta||_1 <= eta
        # soc_2 = [
        #     cp.norm1(beta_k @ pi_k) <= eta_k
        # ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function 
            
        # ( lamda = 2*sqrt(kappa*tau) ) => ( tau=kappa => lambda=2*tau )
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + 2 * np.sqrt(tau*kappa) * cp.norm1(beta_k @ pi_k)
            ),
            soc_1 + wei_sum
        )


        # print("Master problem values")
        # print("tau:", tau)
        # print("kappa:", kappa)
        # print("beta_k:", beta_k)
        # print("X", X )
        # print("X @ beta_k:", X @ beta_k)
        # print("y:", y)
        # print("z_k:", z_k)
        # print("u_k:", u_k)
        

        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        # print('beta_k',beta_k)
        # print('beta_k.shape', beta_k.shape)
        # print('pi_k_sol', pi_k_sol)
        # print('pi_k_sol.shape', pi_k_sol.shape)
        beta_k_sol = beta_k @ pi_k_sol
        # z_k_sol = z_k @ pi_k_sol
        # u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value
        # eta_k_sol = eta_k.value
        

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")

                return None
                
                # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb,  solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                # solved_duals.append(k)
        else:
            return None
            # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            # solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            # if k in solved_duals:
            #     dual_times.append(dual_time)
            #     dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )

        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            # if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol: # OLD
            if np.max(np.abs(lambda_k_diff)) < cg_lambda_tol: # infinity norm < tol

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, None, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                # print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                # print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================
        # print((m, v))
        # Check condition of |psi_k_sol.T @ X| > tau + kappa (-inf inmediately) 
        # It's a best bound than |psi_k_sol.T @ X| > 0. Only possible because of b_i^2 <= z_i*u_i
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > tau + kappa).any() if not dummy_condition else False
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa*tau)).any() if not dummy_condition else False
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau) > -cg_lambda_tol).any() if not dummy_condition else False
        
        if not lagrange_boundedness_conditon:
            # print('condition min diff:')
            # print(np.max(np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau)))

            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")
            print("-"*50)

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            eta = cp.Variable(1, name="eta", nonneg=True)

            # =============================================================================
            #                             Objective Function
            # =============================================================================


            # # 2.2 Cone 2: ||beta||_1 <= eta
            # soc_2_L = [
            #     cp.norm1(beta) <= eta
            # ]

            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    # + xi**2 - mu_k_sol*xi  # irrelevant for this case
                    + (- mu_k_sol/2) * mu_k_sol/2 \
                    + psi_k_sol.T @ X @ beta \
                    +  2 * np.sqrt(tau*kappa) * cp.norm1(beta) \
                    - psi_k_sol.T @ y # Constant (may be removed)
                ),
                # soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    **solver_params
                    # Threads=10,
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            # print lagrangian times
            print('Time elapsed in Lagrange of iteration  6')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)

        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            psi_k_sol_X = psi_k_sol.T @ X
            if psi_k_sol_X.shape[0] == 1:	# If is shape (1,50)
                v_constraint        = psi_k_sol_X[0]
            else:                           # Shape (50,)
                v_constraint        = psi_k_sol_X
            
            # Option 1: n indices
            v_indices           = np.argpartition(np.abs(v_constraint), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)

            # positive beta
            v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
            s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices_pos, s_pos] = 1

            # negative beta
            v_indices_neg = v_indices[v_constraint[v_indices] > 0]
            s_neg = range(len(v_indices_pos),v)#range(len(v_indices_neg))
            beta_sol[v_indices_neg, s_neg] = -1

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            xi_sol     = mu_k_sol/2


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            # if np.sqrt(residuals @ residuals) < cg_residuals_tol: # OLD
            if np.max(np.abs(residuals)) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	


                return beta_final_sol, xi_k_sol, None, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0") # , np.sqrt(residuals @ residuals)

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)

        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)

        k += 1

# LASSO CG SOC2 (l1-norm) relaxation [NOT TRIVIAL. THINK ABOUT IT LONGER. NOT READY TO USE]
def CG_LASSO_SOC2(X,y,tau,kappa, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    # np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        rand_beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        rand_beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        rand_beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        rand_beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, rand_beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    # z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # z_k[beta_k != 0] = 1*np.sqrt(kappa/tau)
    # u_k[beta_k != 0] = 1*np.sqrt(tau/kappa)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        print("-"*50)
        print("Master Problem")
        print("-"*50)

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        eta_k = cp.Variable(1, name="eta_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ beta_k @ pi_k
                )
        ]

        # 2. Cone 2: First order cone of beta: ||beta||_1 <= eta
        soc_2 = [
            cp.norm1(beta_k @ pi_k) <= eta_k
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function 
            
        # ( lamda = 2*sqrt(kappa*tau) ) => ( tau=kappa => lambda=2*tau )
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + 2 * np.sqrt(tau*kappa) * eta_k
            ),
            soc_1 + soc_2 +wei_sum
        )


        # print("Master problem values")
        # print("tau:", tau)
        # print("kappa:", kappa)
        # print("beta_k:", beta_k)
        # print("X", X )
        # print("X @ beta_k:", X @ beta_k)
        # print("y:", y)
        # print("z_k:", z_k)
        # print("u_k:", u_k)
        

        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        # print('beta_k',beta_k)
        # print('beta_k.shape', beta_k.shape)
        # print('pi_k_sol', pi_k_sol)
        # print('pi_k_sol.shape', pi_k_sol.shape)
        beta_k_sol = beta_k @ pi_k_sol
        # z_k_sol = z_k @ pi_k_sol
        # u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value
        # eta_k_sol = eta_k.value
        

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)

            print('SOC2', soc_2[0].dual_value, len(soc_2))
            break

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")

                return None
                
                # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb,  solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
                # solved_duals.append(k)
        else:
            return None
            # psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau, kappa, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            # solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            # if k in solved_duals:
            #     dual_times.append(dual_time)
            #     dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )

        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            # if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol: # OLD
            if np.max(np.abs(lambda_k_diff)) < cg_lambda_tol: # infinity norm < tol

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, None, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                # print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                # print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================
        # print((m, v))
        # Check condition of |psi_k_sol.T @ X| > tau + kappa (-inf inmediately) 
        # It's a best bound than |psi_k_sol.T @ X| > 0. Only possible because of b_i^2 <= z_i*u_i
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > tau + kappa).any() if not dummy_condition else False
        # lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) > 2*np.sqrt(kappa*tau)).any() if not dummy_condition else False
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau) > -cg_lambda_tol).any() if not dummy_condition else False
        
        if not lagrange_boundedness_conditon:
            # print('condition min diff:')
            # print(np.max(np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau)))

            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")
            print("-"*50)

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            eta = cp.Variable(1, name="eta", nonneg=True)

            # =============================================================================
            #                             Objective Function
            # =============================================================================


            # # 2.2 Cone 2: ||beta||_1 <= eta
            # soc_2_L = [
            #     cp.norm1(beta) <= eta
            # ]

            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    # + xi**2 - mu_k_sol*xi  # irrelevant for this case
                    + (- mu_k_sol/2) * mu_k_sol/2 \
                    + psi_k_sol.T @ X @ beta \
                    +  2 * np.sqrt(tau*kappa) * cp.norm1(beta) \
                    - psi_k_sol.T @ y # Constant (may be removed)
                ),
                # soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    **solver_params
                    # Threads=10,
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            # print lagrangian times
            print('Time elapsed in Lagrange of iteration  6')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)

        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            psi_k_sol_X = psi_k_sol.T @ X
            if psi_k_sol_X.shape[0] == 1:	# If is shape (1,50)
                v_constraint        = psi_k_sol_X[0]
            else:                           # Shape (50,)
                v_constraint        = psi_k_sol_X
            
            # Option 1: n indices
            v_indices           = np.argpartition(np.abs(v_constraint), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)

            # positive beta
            v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
            s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices_pos, s_pos] = 1

            # negative beta
            v_indices_neg = v_indices[v_constraint[v_indices] > 0]
            s_neg = range(len(v_indices_pos),v)#range(len(v_indices_neg))
            beta_sol[v_indices_neg, s_neg] = -1

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            xi_sol     = mu_k_sol/2


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb, solver=solver, solver_params=solver_params, solver_verbose=solver_verbose)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            # if np.sqrt(residuals @ residuals) < cg_residuals_tol: # OLD
            if np.max(np.abs(residuals)) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	


                return beta_final_sol, xi_k_sol, None, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0") # , np.sqrt(residuals @ residuals)

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)

        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)

        k += 1


def CG_SOC1_SOC2_upgrade(X,y,tau_tilda,kappa_tilda, eps_soc2_sqrt=0, solver=cp.MOSEK,
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False):
    n,m = X.shape
    v*=2
    # =============================================================================
    #                            SOCP Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)
    # v = 30   # number of solutions to include in each iteration
    # v0 = 5

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    np.random.seed(123)

    # Random initial solution
    rand_beta_k =    np.random.randint(-1,2, size=(m, (v0-1)+k+(v-1)*k_v)).astype(float)  
    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    
    if add_constant:

        # original
        beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

        # new (incluir sol. OLS)

        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # reshape beta_OLs to (m,1)
        # beta_k = np.concatenate((beta_OLS.reshape(-1,1), rand_beta_k), axis=1)
        # v0 +=1

        # # new
        # beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        # beta_k[:,1] = -1 # second column is -1
        # v0 +=1

    else :
        beta_k = rand_beta_k

    # original
    # z_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # u_k =       np.ones((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # new
    z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    z_k[beta_k != 0] = 1
    u_k[beta_k != 0] = 1


    # # Second option for first feasible solution
    # # best_candidates = np.argpartition(beta_OLS, -v)[-v:]
    # best_candidates = [27,  35,  36,  39,  43,  70,  82,  94, 105, 124, 126, 127, 150, 173, 174, 175, 176, 183, 204, 211, 214, 217, 240, 241, 246, 270, 311, 313, 317, 371, 378, 381, 390, 394, 395, 403, 406, 419, 423, 427, 453, 463, 473, 477]
    # noise_size = 10
    # noise = [x for x in range(noise_size) if x not in best_candidates]
    # # print("noise",noise)
    # best_candidates += noise
    # v0 = len(best_candidates)
    # best_candidates_index = [i for i in range(v0)]
    # # beta_k = np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    # beta_k = np.zeros((m,v0))
    # beta_k[best_candidates,best_candidates_index] = 1
    # if add_constant:
    #     beta_k = np.concatenate((np.ones((m,1)), beta_k), axis=1)
    #     v0 +=1
    # z_k = beta_k.copy()
    # u_k = beta_k.copy()
    # print(beta_k)

    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # if save_conv_info:
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Save added columns to the beta_k matrix
    beta_k_added_pos = []
    beta_k_added_neg = []
    beta_k_notadded_pos = [j for j in range(m)]
    beta_k_notadded_neg = [j for j in range(m)]

    # Repeat until convergence
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
        eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                                Master Model
        # =============================================================================

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ (beta_k @ pi_k)
                )
        ]

        # 2. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
        aux_matrix = []
        for i in range(m):
            aux_matrix.append(np.array([( u_k[i,:] - z_k[i,:] ), 2*beta_k[i,:], eps_soc2_sqrt_vector]))
        # print("aux matrix", aux_matrix)
        soc_2 = [
            cp.SOC(
                ( u_k[i,:] + z_k[i,:] ) @ pi_k, # epsilon soc2 is the anti-vertex gap
                aux_matrix[i] @ pi_k
                ) for i in range(m)
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + tau_tilda * cp.sum( z_k @ pi_k ) + kappa_tilda * cp.sum( u_k @ pi_k )
            ),
            soc_1 + soc_2 + wei_sum
        )
        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            # BarConvTol=gp_tol,
            # BarQCPConvTol=gp_tol,
            # Threads=10
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        beta_k_sol = beta_k @ pi_k_sol
        z_k_sol = z_k @ pi_k_sol
        u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)
            soc_2_k_sol = np.array([soc_2[x].dual_value for x in range(len(soc_2))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                # SOC 1
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
                # SOC 2
                soc_2_k_sol = np.array([soc_2[x].dual_value for x in range(len(soc_2))], dtype=object)
                delta_k_sol = soc_2_k_sol[:,0]
                alpha_k_sol = np.zeros(m)
                gamma_k_sol = np.zeros(m)
                for i in range(m):
                    alpha_k_sol[i] = soc_2_k_sol[:,1][i][0]
                    gamma_k_sol[i] = soc_2_k_sol[:,1][i][1]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")
                
                psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb)
                alpha_k_sol = alpha_gamma_k[:,0]
                gamma_k_sol = alpha_gamma_k[:,1]
                solved_duals.append(k)
        else:
            psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb)
            alpha_k_sol = alpha_gamma_k[:,0]
            gamma_k_sol = alpha_gamma_k[:,1]
            solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            if k in solved_duals:
                dual_times.append(dual_time)
                dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )
        lambda_k = np.append(lambda_k, alpha_k_sol)
        lambda_k = np.append(lambda_k, gamma_k_sol )
        lambda_k = np.append(lambda_k, delta_k_sol )


        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol:

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================

        # Check condition of psi_k_sol.T @ X > tau_tilda + kappa_tilda (-inf inmediately)
        # if not (np.abs(psi_k_sol.T @ X) > tau_tilda + kappa_tilda).any():
        # x != 0 <=> |x| > epsilon
        lagrange_boundedness_conditon = (np.abs(psi_k_sol.T @ X - 2*gamma_k_sol) > cg_lambda_tol).any() or \
            (tau_tilda + alpha_k_sol - delta_k_sol < -cg_lambda_tol).any() or \
            (kappa_tilda - alpha_k_sol + delta_k_sol < 0).any() if not dummy_condition else False
        if not lagrange_boundedness_conditon: # [assuming symmetry of tau/kappa]
            # not (kappa_tilda - alpha_k_sol + delta_k_sol < 0).any():# and \ (previous line)

            # =============================================================================
            #                             Lagrangian Model
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")

            # 1. Continuous unbounded
            beta = cp.Variable(m, name="beta", nonneg=False)
            xi =  mu_k_sol/2#cp.Variable(1, name="xi", nonneg=True)

            # 2. Continuous positive 
            z = cp.Variable(m, name="z", nonneg=True)
            u = cp.Variable(m, name="u", nonneg=True)

            # =============================================================================
            #                             Objective Function
            # =============================================================================

            # 1. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    - xi**2 \
                    + (psi_k_sol.T @ X - 2*gamma_k_sol) @ beta \
                    + (tau_tilda + alpha_k_sol - delta_k_sol) @ z \
                    + (kappa_tilda - alpha_k_sol + delta_k_sol) @ u \
                    - psi_k_sol.T @ y # Constant (may be removed)
                )
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            L.solve(
                verbose=solver_verbose, 
                solver=solver,
                **solver_params
                # Threads=10,
                # BarConvTol=gp_tol,
                # BarQCPConvTol=gp_tol,
                )
            
            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)

        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)


        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Two types of sdolution
            beta_cond = (psi_k_sol.T @ X - 2*gamma_k_sol > cg_lambda_tol).any()
            u_z_cond = (tau_tilda + alpha_k_sol - delta_k_sol < -cg_lambda_tol).any()
            if  beta_cond and u_z_cond:
                v1,v2 = v//2, v//2
            elif beta_cond and not u_z_cond:
                v1,v2 = v, 0
            else: # last possible case
                v1,v2 = 0, v

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)
            z_sol       = np.zeros((m, v), dtype=float)
            u_sol       = np.zeros((m, v), dtype=float)

            # Violating constraint
            if  beta_cond:
                psi_k_sol_X_gamma = psi_k_sol.T @ X - 2*gamma_k_sol
                if psi_k_sol_X_gamma.shape[0] == 1:	# If is shape (1,50)
                    v_constraint        = psi_k_sol_X_gamma[0]
                else:                           # Shape (50,)
                    v_constraint        = psi_k_sol_X_gamma
                
                # Option 1: n indices
                v_indices           = np.argpartition(np.abs(v_constraint), -v1)[-v1:]

                # positive beta
                v_indices_pos = v_indices[v_constraint[v_indices] <= 0]
                # print("v1_pos", len(v_indices_pos))
                # print(min(len(v_indices_pos), len(beta_k_notadded_pos)))
                # print("len beta_k_notadded_pos", len(beta_k_notadded_pos))
                # if len(beta_k_notadded_pos) > 0:
                #     if not any([v_ not in beta_k_added_pos for v_ in v_indices_pos]):
                #         min_v1_pos = min(len(v_indices_pos), len(beta_k_notadded_pos))
                #         v_indices_pos[-min_v1_pos:] = np.random.choice(beta_k_notadded_pos, size=min_v1_pos, replace=False)
                beta_k_added_pos += list(v_indices_pos)
                for j_ in v_indices_pos:
                    if j_ in beta_k_notadded_pos:
                        beta_k_notadded_pos.remove(j_)
                s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
                beta_sol[v_indices_pos, s_pos] = 1
                z_sol[v_indices_pos, s_pos] = 1
                u_sol[v_indices_pos, s_pos] = 1

                # negative beta
                v_indices_neg = v_indices[v_constraint[v_indices] > 0]
                # print("v1_neg", len(v_indices_neg))
                # print(min(len(v_indices_neg), len(beta_k_notadded_neg)))
                # print("len beta_k_notadded_neg", len(beta_k_notadded_neg))
                # if len(beta_k_notadded_neg) > 0:
                #     if not any([v_ not in beta_k_added_neg for v_ in v_indices_neg]):
                #         min_v1_neg = min(len(v_indices_neg), len(beta_k_notadded_neg))
                #         v_indices_neg[-min_v1_neg:] = np.random.choice(beta_k_notadded_neg, size=min_v1_neg, replace=False)
                beta_k_added_neg += list(v_indices_neg)
                for j_ in v_indices_neg:
                    if j_ in beta_k_notadded_neg:
                        beta_k_notadded_neg.remove(j_)
                s_neg = range(len(v_indices_pos),v1)#range(len(v_indices_neg))
                beta_sol[v_indices_neg, s_neg] = -1
                z_sol[v_indices_neg, s_neg] = 1
                u_sol[v_indices_neg, s_neg] = 1

            elif u_z_cond:
                alpha_delta_constr = tau_tilda + alpha_k_sol - delta_k_sol
                if alpha_delta_constr.shape[0] == 1:	# If is shape (1,50)
                    v_constraint        = alpha_delta_constr[0]
                else:                           # Shape (50,)
                    v_constraint        = alpha_delta_constr

                # Option 1: n indices
                v_indices           = np.argpartition(v_constraint, v2)[:v2] # we want the smallest (most negative)

                # Sign of beta
                psi_k_sol_X_gamma = psi_k_sol.T @ X - 2*gamma_k_sol
                if psi_k_sol_X_gamma.shape[0] == 1:	# If is shape (1,50)
                    v_constraint2        = psi_k_sol_X_gamma[0]
                else:                           # Shape (50,)
                    v_constraint2        = psi_k_sol_X_gamma
                
                # positive beta
                v_indices_pos = v_indices[v_constraint2[v_indices] <= 0]
                s_pos = [v1+s for s in range(len(v_indices_pos))] #range(len(v_indices_neg),v)
                beta_sol[v_indices_pos, s_pos] = 1
                z_sol[v_indices_pos, s_pos] = 1
                u_sol[v_indices_pos, s_pos] = 1

                # negative beta
                v_indices_neg = v_indices[v_constraint2[v_indices] > 0]
                s_neg = [v1+s for s in range(len(v_indices_pos),v2)]#range(len(v_indices_neg))
                beta_sol[v_indices_neg, s_neg] = -1
                z_sol[v_indices_neg, s_neg] = 1
                u_sol[v_indices_neg, s_neg] = 1

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        else:
            beta_sol   = beta.value
            z_sol      = z.value
            u_sol      = u.value
            xi_sol     = xi


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            if np.sqrt(residuals @ residuals) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # add the following to the return final dict
                # master_values = []
                # lagrangian_values = []
                # dual_values = []

                # # solver times
                # master_times = []
                # lagrangian_times = []
                # dual_times = []

                # # solver p_times
                # master_p_times = []
                # lagrangian_p_times = []
                # dual_p_times = []
                # solved_duals = []
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0", np.sqrt(residuals @ residuals))

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)
            z_k = np.append(z_k, z_sol.reshape(m,1), axis=1)
            u_k = np.append(u_k, u_sol.reshape(m,1), axis=1)
        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)
            z_k = np.append(z_k, z_sol, axis=1)
            u_k = np.append(u_k, u_sol, axis=1)

        k += 1

# soc2: Quizás sacar un equivalente de v_soc2 = n*v_soc1, ya que demora más (y se agrega la mitad de soluciones realmente, con dos signos)
def CG_SOC2(X,y,tau_tilda,kappa_tilda, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False):
    n,m = X.shape
    v*=2
    # =============================================================================
    #                            SOCP 2 Column Generation
    # =============================================================================

    # beta FOC parameters
    inv_X_t_X = np.linalg.inv(X.T @ X)
    beta_OLS = inv_X_t_X @ (X.T @ y) 

    # =============================================================================
    #                               Algorithm
    # =============================================================================
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)
    # v = 30   # number of solutions to include in each iteration
    # v0 = 5

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    np.random.seed(123)

    # Random initial solution
    rand_beta_k =    np.random.randint(-1,2, size=(m, (v0-1)+k+(v-1)*k_v)).astype(float)  
    # rand_beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', rand_beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # rand_beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    
    if add_constant:

        # original
        beta_k = np.concatenate((np.ones((m,1)), rand_beta_k), axis=1)
        v0 +=1

    else :
        beta_k = rand_beta_k

    z_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    u_k =       np.zeros((m, (v0-1)+k+(v-1)*k_v), dtype=float)
    z_k[beta_k != 0] = 1
    u_k[beta_k != 0] = 1


    # First vector of dual variables
    lambda_k_1 = np.array([None])

    # # Beta synthetic solutions 
    # v_indices_all = {}

    # Save convergence info (optmimums)
    # if save_conv_info:
    # optimal values
    master_values = []
    lagrangian_values = []
    dual_values = []

    # solver times
    master_times = []
    lagrangian_times = []
    dual_times = []

    # solver p_times
    master_p_times = []
    lagrangian_p_times = []
    dual_p_times = []
    solved_duals = []

    t0_cg = time()
    t0p_cg = process_time()

    # Repeat until convergence
    added_indices = {}
    while True:

        # Time limit
        if time_limit is not None:
            if (time() - t0_cg)/60 > time_limit:
                print("Time limit reached")
                break

        print("\n","-"*10,"\n",f"ITERATION {k}")

        check_residuals = True # Check residuals only when not creating a virtual solution

        if eps_soc2_sqrt != 0:
            eps_soc2_sqrt_vector = np.zeros((v0-1)+k+(v-1)*k_v, dtype=float)
            eps_soc2_sqrt_vector[:] = eps_soc2_sqrt
    
        # =============================================================================
        #                          Model: Master Problem
        # =============================================================================

        # =============================================================================
        #                             Model Variables
        # =============================================================================
        
        # 1. Continuous unbounded
        # pi_k = cp.Variable(k+v-1, name="pi_k") # 4+k variables bc of the first iteration (5 solutions)
        pi_k = cp.Variable((v0-1)+k+(v-1)*k_v, name="pi_k") # (v0-1)+k+(v-1)*k_v variables bc of the first iteration and v_sols (5 solutions)
        xi_k = cp.Variable(1, name="xi_k")
        
        # =============================================================================
        #                             Model Constraints
        # =============================================================================

        # 1. Cone 1: Linnearization of the residuals norm
        soc_1 = [
            cp.SOC(
                xi_k, 
                y - X @ (beta_k @ pi_k)
                )
        ]

        # 2. Cone 2_i: Conic form of the beta_i <= z_i * u_i constraint 
        aux_matrix = []
        if eps_soc2_sqrt != 0:
            for i in range(m):
                aux_matrix.append(np.array([( u_k[i,:] - z_k[i,:] ), 2*beta_k[i,:], eps_soc2_sqrt_vector]))
        else:
            for i in range(m):
                aux_matrix.append(np.array([( u_k[i,:] - z_k[i,:] ), 2*beta_k[i,:]]))
        # print("aux matrix", aux_matrix)
        soc_2 = [
            cp.SOC(
                ( u_k[i,:] + z_k[i,:] ) @ pi_k, # epsilon soc2 is the anti-vertex gap (appears to be useless)
                aux_matrix[i] @ pi_k
                ) for i in range(m)
        ]

        # 3. Weights must sum 1 and be positive
        wei_sum = []

        if sum_1_comb:
            wei_sum+=[
                cp.sum(pi_k) == 1,    #(This constraint is not necessary maybe?)
            ] 
        
        if pos_linear_comb:
            wei_sum+=[
                pi_k >= 0
            ]

        # =============================================================================
        #                             Objective Function
        # =============================================================================

        # 2. Objective function
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + tau_tilda * cp.sum( z_k @ pi_k ) + kappa_tilda * cp.sum( u_k @ pi_k )
            ),
            soc_1 + soc_2 + wei_sum
        )
        
        # =============================================================================
        #                                Solver
        # =============================================================================     
        
        t0 = time()
        t0p = process_time()
        
        socp.solve(
            verbose=solver_verbose, 
            solver=solver, 
            **solver_params
            # warm_start=True, 
            # BarConvTol=gp_tol,
            # BarQCPConvTol=gp_tol,
            # Threads=10
            )
        
        t1 = time()
        t1p = process_time()


        print("The optimal value is", socp.value)
        print("Time elapsed in iteration ", k)
        print((t1-t0)/60, " mins (normal)")
        print((t1p-t0p)/60, " mins (process)")

        # Add optimum info
        if save_conv_info:
            master_values.append(socp.value)
            master_times.append((t1-t0)/60)
            master_p_times.append((t1p-t0p)/60)


        # =============================================================================
        #                             Solutions 
        # =============================================================================

        # 1. Primal solutions
        pi_k_sol = pi_k.value
        beta_k_sol = beta_k @ pi_k_sol
        z_k_sol = z_k @ pi_k_sol
        u_k_sol = u_k @ pi_k_sol
        xi_k_sol = xi_k.value

        # 2. Dual variable 
        if not solve_dual_directly:
            soc_1_k_sol = np.array([soc_1[x].dual_value for x in range(len(soc_1))], dtype=object)
            soc_2_k_sol = np.array([soc_2[x].dual_value for x in range(len(soc_2))], dtype=object)

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol.any(): 
                # SOC 1
                psi_k_sol = soc_1_k_sol[0][1]
                mu_k_sol = soc_1_k_sol[0][0][0]
                # SOC 2
                delta_k_sol = soc_2_k_sol[:,0]
                alpha_k_sol = np.zeros(m)
                gamma_k_sol = np.zeros(m)
                for i in range(m):
                    alpha_k_sol[i] = soc_2_k_sol[:,1][i][0]
                    gamma_k_sol[i] = soc_2_k_sol[:,1][i][1]
            else:
                print("Primal-Dual gap is too big, no dual solution")
                print("Solving dual problem... (directly)")
                
                psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb)
                alpha_k_sol = alpha_gamma_k[:,0]
                gamma_k_sol = alpha_gamma_k[:,1]
                solved_duals.append(k)
        else:
            psi_k_sol, mu_k_sol, alpha_gamma_k, delta_k_sol, dual_time, dual_p_time  = masters_dual(X, y, tau_tilda, kappa_tilda, beta_k, z_k, u_k, k, v, k_v, v0, sum_1_comb)
            alpha_k_sol = alpha_gamma_k[:,0]
            gamma_k_sol = alpha_gamma_k[:,1]
            solved_duals.append(k)

        # Save convergence info
        if save_conv_info:
            dual_values.append(- psi_k_sol.T @ y)

            if k in solved_duals:
                dual_times.append(dual_time)
                dual_p_times.append(dual_p_time)

        #==============================================================================
        #                             First Stopping Criterion
        #==============================================================================

        # Aggregation of the dual variables in a single vector
        lambda_k = np.append(psi_k_sol, mu_k_sol )


        # Check if lamda_k is equal to lambda_k_1. If so, stop the algorithm.
        # If not, continue the algorithm. Then, update the lambda_k_1 with the lambda_k.
        if k > 1:
            lambda_k_diff = lambda_k - lambda_k_1
            if np.sqrt(lambda_k_diff @ lambda_k_diff) < cg_lambda_tol:

                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()

            # if np.array_equal(lambda_k, lambda_k_1):
                print("First stopping criterion met: lambda_k = lambda_k_1")
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate:", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
                
            else:
                print("First stopping criterion NOT met: lambda_k != lambda_k_1")
                print("Lambda diff:", lambda_k_diff @ lambda_k_diff)
                print("Lambda diff sqrt:", np.sqrt(lambda_k_diff @ lambda_k_diff))

        lambda_k_1 = lambda_k

        #==============================================================================
        #                             Second Stopping Criterion
        #==============================================================================

        # Check condition of (tau + alpha - gamma) < 0  (-inf inmediately) [case: symmetric kappa/tau]
        lagrange_boundedness_conditon = ((tau_tilda + alpha_k_sol - gamma_k_sol < -cg_lambda_tol).any() or \
            (kappa_tilda - alpha_k_sol - gamma_k_sol < 0).any()) if not dummy_condition else False
        if not lagrange_boundedness_conditon: #+ alpha_k_sol
            # and not (kappa_tilda - alpha_k_sol - gamma_k_sol < 0).any():

            # =============================================================================
            #                             Lagrangian Model (just for u and z)
            # =============================================================================
            
            print("-"*50)
            print("Lagrangian Model")

            # 1. Continuous unbounded
            # beta = cp.Variable(m, name="beta", nonneg=False)
            # xi = cp.Variable(1, name="xi", nonneg=True)

            # 2. Continuous positive 
            z = cp.Variable(m, name="z", nonneg=True)
            u = cp.Variable(m, name="u", nonneg=True)

            # 3. Aux vector
            # aux_vector = cp.Variable((m, 2 + (eps_soc2_sqrt_L!=0)), name="aux_vector", nonneg=False, boolean=False, integer=False)

            # =============================================================================
            #                             Objective Function
            # =============================================================================

            # 0. First Order Condition for beta: When tau == kappa
            # beta := np.linalg.inv(X.T @ X) @ (X.T @ y + gamma_k_sol) 
            beta = beta_OLS + inv_X_t_X @ gamma_k_sol
            xi = np.sqrt((y - X@beta).T @ (y - X@beta))


            # 2. Objective function
            L = cp.Problem(
                cp.Minimize(
                    0 \
                    + xi**2 # irrelevant for this case
                    - 2 * gamma_k_sol @ beta \
                    + (tau_tilda + alpha_k_sol - delta_k_sol) @ z \
                    + (kappa_tilda - alpha_k_sol - delta_k_sol) @ u # no constant term
                ),
                # aux_constr + soc_2_L
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            L.solve(
                verbose=solver_verbose, 
                solver=solver,
                **solver_params
                # Threads=10,
                # BarConvTol=gp_tol,
                # BarQCPConvTol=gp_tol,
                )
            
            t1 = time()
            t1p = process_time()

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)

            print('-'*50)
            print("The Lagrangian optimal value is", L.value)


        # Dummy skip
        else:
            print("Skipping Lagrangian Model")
            L = Dummy()

            if save_conv_info:
                lagrangian_values.append(L.value)
                lagrangian_times.append(None)
                lagrangian_p_times.append(None)


        # If it is -inffinity, then check the betas dual constraints
        if L.value == float('-inf') or L.value is None:

            print("Lagrangian is -inffinity")

            # print("Breaking the algorithm")
            print("Checking the dual constraints...")

            # Violating constraint
            tau_alpha_delta = tau_tilda + alpha_k_sol - delta_k_sol
            if tau_alpha_delta.shape[0] == 1:	# If is shape (1,m)
                v_constraint        = tau_alpha_delta[0]
            else:                           # Shape (m,)
                v_constraint        = tau_alpha_delta

            # gamma_beta = -2*gamma_k_sol
            # if gamma_beta.shape[0] == 1:	# If is shape (1,m)
            #     v_constraint        = gamma_beta[0]
            # else:                           # Shape (m,)
            #     v_constraint        = gamma_beta
            
            # Option 1: v least indices 
            # print('antes los ignorantes eran como más humildes, weon')
            # print(v_constraint[v_constraint < 0])
            v_indices           = np.argpartition(v_constraint, v//2+v%2)[:v//2+v%2] # original
            s_all = range(len(v_indices))
            # v_indices           = np.argpartition(np.abs(beta), -v)[-v:]

            # adding solutions (canonical)
            # Option 1
            beta_sol    = np.zeros((m, v), dtype=float)
            z_sol       = np.zeros((m, v), dtype=float)
            u_sol       = np.zeros((m, v), dtype=float)


            # Original
            # positive beta
            # v_indices_pos = v_indices[beta[v_indices] <= 0]
            # s_pos = range(len(v_indices_pos))#range(len(v_indices_neg),v)
            beta_sol[v_indices, s_all] = 1
            z_sol[v_indices, s_all] = 1
            u_sol[v_indices, s_all] = 1

            # negative beta
            # v_indices_neg = v_indices[beta[v_indices] > 0]
            s_neg = range(len(v_indices),v)#range(len(v_indices_neg))
            beta_sol[v_indices[:v_indices.size-v%2], s_neg] = -1
            z_sol[v_indices[:v_indices.size-v%2], s_neg] = 1
            u_sol[v_indices[:v_indices.size-v%2], s_neg] = 1

            # Update k_v
            k_v += 1

            # Virtual solution, so we are not checking residuals
            check_residuals = False

        
        else: # (L converged to a finite value)
            # beta_sol   = beta.value
            # z_sol      = z.value
            # u_sol      = u.value
            # xi_sol     = xi.value

            beta_sol   = beta
            z_sol      = z.value
            u_sol      = u.value
            xi_sol     = xi


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # # V1: Matrix of the matrices beta_k, z_k, u_k
            # x_beta_z_u = np.zeros((3*m,k+v-1))
            # x_beta_z_u = np.zeros((3*m,(v0-1)+k+(v-1)*k_v))
            # x_beta_z_u[:m,:] += beta_k
            # x_beta_z_u[m:2*m,:] += z_k
            # x_beta_z_u[2*m:,:] += u_k

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

            # # V1: Objective: the actual solution of the Lagrangian model
            # y_beta_z_u = np.append(beta_sol, z_sol)
            # y_beta_z_u = np.append(y_beta_z_u, u_sol)

            # V2: Objective: the actual solution of the Lagrangian model for beta
            y_beta_z_u = beta_sol.copy()


            # NORMALIZATION of the parameters of the regression (sum of coef = 1)
            if sum_1_comb:
                pi_hat, residuals = convex_linnear_regression(x_beta_z_u, y_beta_z_u, pos_linear_comb)
            else:
                # OLS solving the linear system. Then get the residuals
                lm = LinearRegression(fit_intercept=False, n_jobs=-1, positive=pos_linear_comb)
                lm.fit(x_beta_z_u, y_beta_z_u)
                pi_hat = lm.coef_
                residuals = y_beta_z_u - x_beta_z_u @ pi_hat

            # # Prediction of the solution
            # residuals = y_beta_z_u - x_beta_z_u @ pi_hat  # new_sol - ols_sol

            # Condition to stop the algorithm
            if np.sqrt(residuals @ residuals) < cg_residuals_tol:
                print('-'*50)
                t1_cg = time()
                t1p_cg = process_time()
                print("Second stopping criterion met: residuals**2 == 0:", np.sqrt(residuals @ residuals))
                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                # print(beta_final_sol)
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)
                # print(y_final_sol)

                # # Plot real vs predicted
                # plt.figure(figsize=(6,6))
                # plt.scatter(y, y_final_sol, c='C1')
                # plt.title(f"y v/s y_hat (Original)\nn_col={n_col}, n_row={n_row}")
                # plt.xlabel("y")
                # plt.ylabel("y_hat")
                # plt.show()

                # print("#Betas sobrev.:", np.sum(u_k_sol > pos_tol), "(rate = ", np.sum(u_k_sol > pos_tol)/m, ")")
                # print("Betas sobrev.:", np.where(u_k_sol > pos_tol)[0])
                # print("Error cuadratico:", error_quad)	
                # print("Error medio:", np.sqrt(error_quad))	

                # add the following to the return final dict
                # master_values = []
                # lagrangian_values = []
                # dual_values = []

                # # solver times
                # master_times = []
                # lagrangian_times = []
                # dual_times = []

                # # solver p_times
                # master_p_times = []
                # lagrangian_p_times = []
                # dual_p_times = []
                # solved_duals = []
                return beta_final_sol, xi_k_sol, u_k_sol, z_k_sol, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
            print("Second stopping criterion NOT met: residuals**2 != 0", np.sqrt(residuals @ residuals))

        # ==============================================================================
        #                       Keep the loop going (updates)
        # ==============================================================================

        # 3. K iteration of the solutions. Add the (beta, z, u) weighted sum with the pi solutions to the 
        # respective matices.

        # 3.1. If beta_sol is a vector, then add it to the matrix
        if beta_sol.ndim == 1:
            beta_k = np.append(beta_k, beta_sol.reshape(m,1), axis=1)
            z_k = np.append(z_k, z_sol.reshape(m,1), axis=1)
            u_k = np.append(u_k, u_sol.reshape(m,1), axis=1)
        # 3.2. If beta_sol is a matrix, then add all the columns to the matrix
        else:
            beta_k = np.append(beta_k, beta_sol, axis=1)
            z_k = np.append(z_k, z_sol, axis=1)
            u_k = np.append(u_k, u_sol, axis=1)

        k += 1


# No se puede hacer CG_SOC2 así como así. No existe relación entre beta y z,y; sólo entre beta y xi, o entre los términos mismos de u y z.
# Para u,z divirgiendo, se puede buscar el signo negativo más grande de sus coeficientes (con los que se quieren ir a -inf) para hacer las
# soluciones sintéticas, pero para (xi - gamma'beta) no tenemos un valor de xi previo para obtener índices de sol. sintética.
# Como no podemos armar la sol sintética entera, entonces no podemos hacer CG_SOC2.


# %%

#%%
        

# %%
