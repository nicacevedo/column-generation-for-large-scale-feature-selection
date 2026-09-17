# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# COMPATIBILITY COPY of `src/cg_models.py` (2025-06-07, commit 58d8b62).
#
# The historical code does not run on this machine. It was written against
# numpy 1.x / cvxpy ~1.3 with MOSEK; this host has numpy 2.5.3, cvxpy 1.9.2 and
# no MOSEK licence. Rather than rewrite the method -- a rewrite is not a
# reproduction -- this is a byte-for-byte copy with the smallest diff that
# makes it execute, so that `git diff src/cg_models.py src/cg2026/cg_hist_compat.py`
# is the exact and complete record of what had to change.
#
# THE WHOLE DIFF, and why each line is not a change to the algorithm:
#
# 1. Dual extraction. `soc_1[x].dual_value` is a *list* `[mu, psi]` of ragged
#    arrays in cvxpy 1.9. `np.array([...], dtype=object)` on it raises
#    `ValueError: setting an array element with a sequence` under numpy 2, and
#    `.any()` never runs. Replaced with a plain list and an explicit
#    `is not None` test. Same two quantities, same order, no arithmetic.
#
# 2. Plotting is stubbed out. The historical code plots its bound history on
#    every termination path, and that code has a pre-existing off-by-one
#    (`range(1, k)`, which is k-1 points, against k recorded values) that raises
#    whenever the path is taken with k >= 4. Disabled rather than fixed,
#    because fixing it would be changing the historical code. No scientific
#    value is computed from a plot.
#
# Nothing else. The master, the pricing, the boundedness test, the tolerances,
# the stopping criteria, the column construction and every default argument are
# the 2025 code unchanged.
#
# Any result produced with this file is a PARTIAL_REPRODUCTION: original code,
# original parameters, a different conic solver (Clarabel, because MOSEK is not
# licensed here) and a 2026 numerical stack. It is never an
# ORIGINAL_REPRODUCTION and is never compared against a historical MOSEK timing
# as though it were the same measurement.
# ---------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

# Copy of "models.py", created in 21-04-2025 to continue with the research

"""
Spyder Editor

This is a temporary script file.
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "10"
import cvxpy as cp

import numpy as np

# COMPAT (2): plotting is disabled, not fixed.
#
# The historical code draws its bound history on every termination path, with
# `iterations = list(range(1, k))` -- k-1 points -- against `master_values`,
# which has k. matplotlib raises `x and y must have same first dimension`
# whenever that path is taken with k >= 4. It is a pre-existing defect and it
# is consistent with the 2025 notes' own transcript, which ends at "Time limit
# reached / Plotting bounds..." and nothing after.
#
# Fixing it would be changing the historical code. No scientific value is
# computed from a plot, so the module is replaced with a stub that accepts
# everything and does nothing.
class _NoPlot:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


plt = _NoPlot()

from time import time, process_time

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, LassoLars 
from scipy.optimize import minimize

# from model_solver import GUROBI_MODEL


def keep_top_k(arr, k):
    # Make a copy to avoid modifying the original array
    out = np.zeros_like(arr)
    if k <= 0:
        return out
    # Find indices of the k largest values
    idx = np.argpartition(arr, -k)[-k:]
    # Set only those positions to their original values
    out[idx] = arr[idx]
    return out


# Random seed
np.random.seed(123)


# Dummy auxiliar class
class Dummy:
    def __init__(self) -> None:
        self.value = None


# LASSO CG with f.o. l1 instead of constraint. (Almost the same results as v1)
def CG_LASSO_SOC1_v2(X,y,tau,kappa, eps_soc2_sqrt=0, solver=cp.MOSEK, 
    solver_params={'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-6}}, solver_verbose=False, 
    eps_soc2_sqrt_L=0, add_constant=True, save_conv_info=False, sum_1_comb=False, pos_linear_comb=False,
    solve_dual_directly=False, cg_lambda_tol=1e-6, cg_residuals_tol=1e-6,
    v=5, v0=0, time_limit=None, dummy_condition=False, canonic_random_initial_sol=True,
    # 2025 inputs
    unboundedness_policy="v_solution" # negative_gradient, unit_ball, mixture
    ):
    n,m = X.shape
    # =============================================================================
    #                            SOCP 1 Column Generation
    # =============================================================================

    # =============================================================================
    #                               Algorithm
    # =============================================================================

    print("""
    # =============================================================================
    #                             LASSO SOC1 CG v2
    # =============================================================================
          """)
    assert unboundedness_policy in ["v_solution", "unit_ball", "negative_gradient", "mixture"]
    
    # # Iteration and solutions to include in each iteration
    k = 1   # itaration index
    k_v = 0 # itaration index of the multiple solutions (v_sols)

    # First feasible solution (beta, z, u) with 3 * m components
    # (**For now: beta in {0,1}^m and z = u = 1 in R^m)
    # seed=np.random.randint(1,999999)
    np.random.seed(123)

    # Random initial solution
    if not canonic_random_initial_sol:
        beta_k =    np.random.randint(-1,2, size=(m, v0)).astype(float)  
        # Random initial solution: generate a matriz of size=(m, (v0-1)+k+(v-1)*k_v),
        # with (v0-1)+k+(v-1)*k_v random canonic vector of R^m as columns

    else:
        # canonic random solutions
        beta_k =   np.zeros((m, v0), dtype=float)
        v0_positive = np.random.randint(0, v0 + 1)
        # print(m, v0, v0_positive)
        rand_beta_k_indices_pos = np.random.choice(m, size= v0_positive, replace=False)
        rand_beta_k_indices_neg = np.random.choice(m, size= v0 - v0_positive, replace=False)
        beta_k[rand_beta_k_indices_pos, np.arange(v0_positive)] = 1#np.random.choice([-1,1], size=(v0-1)+k+(v-1)*k_v, replace=True)
        beta_k[rand_beta_k_indices_neg, np.arange(v0_positive, v0)] = -1

    # beta_k =    np.zeros((m, (v0-1)+k+(v-1)*k_v)).astype(float) # Guardar el bicho en otra variable
    # print('beta shape', beta_k.shape)
    # random_init_indices = np.random.choice(m, size=(v0-1)+k+(v-1)*k_v, replace=False)
    # beta_k[random_init_indices, np.arange((v0-1)+k+(v-1)*k_v)] = 1
    if add_constant:

        # original
        constant_column = np.ones((m,1), dtype=float)
        # constant_column = np.random.choice([-1,1], size=(m,1), replace=True) #np.ones((m,1), dtype=float)
        beta_k = np.concatenate((constant_column, beta_k), axis=1)
        # beta_k = np.concatenate((np.ones((m,1)), beta_k), axis=1)
        v0 +=1


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
        if time_limit:
            if (time() - t0_cg)/60 > time_limit:
                print('-'*50)
                print("TIME LIMIT REACHED")
                t1_cg = time()
                t1p_cg = process_time()

                print("Total time elapsed in CG")
                print((t1_cg-t0_cg)/60, " mins (normal)")
                print((t1p_cg-t0p_cg)/60, " mins (process)")

                print("Plotting bounds...")
                
                # Example data
                iterations = list(range(1, k))
                lower_bounds = lagrangian_values
                upper_bounds = master_values
                print("UB:", upper_bounds)
                print("LB:", lower_bounds)

                # Plot
                k_init = 2
                plt.plot(iterations[k_init:], upper_bounds[k_init:], label='Upper Bound', marker='s')
                plt.plot(iterations[k_init:], lower_bounds[k_init:], label='Lower Bound', marker='o')
                plt.xlabel('Iteration')
                plt.ylabel('Bound Value')
                plt.title('Bounds per Iteration')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.yscale("log")
                plt.show()

                # Solution of the algorithm
                beta_final_sol = beta_k_sol
                y_final_sol = X @ beta_final_sol
                error_quad = (y_final_sol - y) @ (y_final_sol - y)

                # return beta, xi, u, z, aux_vector, socp, (t1-t0)/60, (t1p-t0p)/60
                return beta_final_sol, xi_k_sol, beta_final_sol, beta_final_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
                    'k':k, 'k_v':k_v, 'master_values':master_values, 'lagrangian_values':lagrangian_values, 'dual_values':dual_values, 
                    'master_times':master_times, 'lagrangian_times':lagrangian_times, 'dual_times':dual_times,
                    'master_p_times':master_p_times, 'lagrangian_p_times':lagrangian_p_times, 'dual_p_times':dual_p_times,
                    'solved_duals':solved_duals
                    }
 


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
        eta_k = cp.Variable(m, name="eta_k")
        
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
        soc_2 = [
            beta_k @ pi_k <= eta_k,
            -beta_k @ pi_k <= eta_k
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
        e_m = np.ones(m)
        socp = cp.Problem(
            cp.Minimize(
                xi_k**2 + 2 * np.sqrt(tau*kappa) * e_m @ eta_k#cp.norm1(beta_k @ pi_k)
            ),
            soc_1 + wei_sum + soc_2
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
            warm_start=True,
            **solver_params
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
            # COMPAT (1): a plain list, and an explicit None test. The original
            # `np.array([...], dtype=object)` raises under numpy 2 because the
            # per-constraint dual value is itself a ragged list [mu, psi].
            soc_1_k_sol = [soc_1[x].dual_value for x in range(len(soc_1))]

            # If it can compute the dual variable, it will be stored in the lambda_k_1 vector
            if soc_1_k_sol and soc_1_k_sol[0] is not None: 
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


                
                print("Plotting bounds...")
                
                # Example data
                iterations = list(range(1, k))
                lower_bounds = lagrangian_values
                upper_bounds = master_values
                print("UB:", upper_bounds)
                print("LB:", lower_bounds)

                # Plot
                k_init = 2
                plt.plot(iterations[k_init:], upper_bounds[k_init:], label='Upper Bound', marker='s')
                plt.plot(iterations[k_init:], lower_bounds[k_init:], label='Lower Bound', marker='o')
                plt.xlabel('Iteration')
                plt.ylabel('Bound Value')
                plt.title('Bounds per Iteration')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.yscale("log")
                plt.show()

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
                return beta_final_sol, xi_k_sol, beta_final_sol, beta_final_sol, socp.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
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
        # unbounded_lagrangian = (np.abs(psi_k_sol.T @ X) > tau + kappa).any() if not dummy_condition else False
        # unbounded_lagrangian = (np.abs(psi_k_sol.T @ X) >= 2*np.sqrt(kappa*tau)).any() + cg_lambda_tol if not dummy_condition else False
        unbounded_lagrangian = (np.abs(psi_k_sol.T @ X) - 2*np.sqrt(kappa*tau) > -cg_lambda_tol).any() if not dummy_condition else False
        # print("unbounded lagrangian?: ", unbounded_lagrangian)
        # print("unbounded and uni ball: ", unbounded_lagrangian and unboundedness_policy == "unit_ball")
        if not unbounded_lagrangian or (unbounded_lagrangian and unboundedness_policy == "unit_ball") or (unbounded_lagrangian and unboundedness_policy == "mixture" and k%2 == 0):
            print("entering the lagrangian problem...")
            # exit()
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
            # eta = cp.Variable(1, name="eta", nonneg=True)

            # =============================================================================
            #                             Objective Function
            # =============================================================================

            unit_ball = []
            if (unbounded_lagrangian and unboundedness_policy == "unit_ball") or (unbounded_lagrangian and unboundedness_policy == "mixture" and k%2 == 0):
                # Unit ball constaint over beta
                unit_ball += [
                    cp.SOC(
                        1, 
                        beta
                        )
                ]   

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
                unit_ball
            )
            
            # =============================================================================
            #                               Solver
            # =============================================================================
            
            t0 = time()
            t0p = process_time()
            
            try:
                # l_solver_params = solver_params.copy()
                # l_solver_params['mosek_params']['MSK_DPAR_INTPNT_CO_TOL_REL_GAP'] = cg_lambda_tol * 1e-2
                # print(l_solver_params)
                L.solve(
                    verbose=solver_verbose, 
                    solver=solver,
                    warm_start=True,
                    **solver_params
                    )
            except Exception as e:
                # Dummy skip
                print("Forcing skip in Lagrangian Model")
                L = Dummy()
                # L value equal to -inf
                L.value = float('-inf')

            t1 = time()
            t1p = process_time()

            # print lagrangian times
            print(f'Time elapsed in Lagrange of iteration {k}')
            print((t1 - t0)/60, 'mins (normal)')
            print((t1p - t0p)/60, 'mins (process)')

            if (unbounded_lagrangian and unboundedness_policy == "unit_ball") or (unbounded_lagrangian and unboundedness_policy == "mixture" and k%2 == 0):

                # lambda_1 = np.sqrt(tau * kappa)
                # psi_k_sol_X = (psi_k_sol.T @ X)[0]
                # print(psi_k_sol_X.shape)
                # unbounded_indices = np.abs(psi_k_sol_X) > lambda_1 + cg_lambda_tol
                # print("unbounded_indices: ", unbounded_indices.shape)
                # print("beta: ", beta.value.shape)
                # minus_grad = np.zeros(beta.value.size)
                # print("beta selected: ", beta_k_sol[unbounded_indices])
                # minus_grad[unbounded_indices] = - ( psi_k_sol_X[unbounded_indices] - lambda_1 * beta_k_sol[unbounded_indices]/np.abs(beta_k_sol[unbounded_indices]) ) # beta is the opposite sign as psi  
                # print("\nMinus grad value direction (unbounded coordinates):\n")
                # print(minus_grad)
                # print("\nMinus grad value direction (unbounded coordinates) normalized:\n")
                # print(minus_grad/np.linalg.norm(minus_grad))
                beta_value = (beta.value).copy()
                # zero-value cleaning & normalization
                # print("\nBeta value from unit-ball:\n")
                # print(beta_value)
                print("Re-normalizing direction of unit ball and re-define inf L.value...")
                beta_value[np.abs(beta_value) <= cg_lambda_tol] = 0
                beta_value = beta_value / np.max(np.abs(beta_value)) # -1/1 interval normalization
                # beta_value = beta_value / np.max(np.abs(beta_value))
                beta_value[np.abs(beta_value) <= cg_lambda_tol] = 0
                # beta_value = keep_top_k(beta_value, k=v)
                # print("\nBeta value from unit-ball after clearning & re-scaling:\n")

                # print(beta_value)

                # exit()

                # # Redefininf objective value also
                # L_value =  (- mu_k_sol/2) * mu_k_sol/2 \
                #     + psi_k_sol.T @ X @ beta_value \
                #     +  2 * np.sqrt(tau*kappa) * cp.norm1(beta_value) \
                #     - psi_k_sol.T @ y # Constant (may be removed)
                # L_value = L_value.value[0]

                L_value = L.value

                
            else:
                L_value = L.value

            print('-'*50)
            print("The Lagrangian original optimal value is", L.value)
            print("The Lagrangian normalized optimal value is", L_value)

            # Save convergence optimum
            if save_conv_info:
                lagrangian_values.append(L_value)
                lagrangian_times.append((t1-t0)/60)
                lagrangian_p_times.append((t1p-t0p)/60)


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

            if (unboundedness_policy == "v_solution") or (unboundedness_policy == "mixture" and k%2 == 1):
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
            elif unboundedness_policy == "negative_gradient":
                lambda_1 = 2 * np.sqrt(tau*kappa) 
                psi_k_sol_X = (psi_k_sol.T @ X).copy().reshape(beta_k_sol.shape)
                divergent_indices = np.abs(psi_k_sol_X) > lambda_1
                # print("divergent indices: ", divergent_indices)
                # convergent_indices = np.abs(psi_k_sol_X) <= lambda_1
                beta_term = np.zeros(beta_k_sol.size) # ALTERNATIVE
                # psi_k_sol_X[psi_k_sol_X!=0] = psi_k_sol_X[psi_k_sol_X!=0]/np.abs(psi_k_sol_X[psi_k_sol_X!=0]) # gradient term normalizaton     
                # print(psi_k_sol_X[divergent_indices])    
                # print(psi_k_sol_X[divergent_indices])      
                beta_term[divergent_indices] = lambda_1 * psi_k_sol_X[divergent_indices]/np.abs(psi_k_sol_X[divergent_indices]) - psi_k_sol_X[divergent_indices] #).reshape(beta_k_sol.shape)
                beta_term[np.abs(psi_k_sol_X) <= cg_lambda_tol] = 0 # get rid of almost-zero terms
                
                # beta_sol[convergent_indices] = 0 # get rid of convergent coordinates
                beta_sol = beta_term / np.max(np.abs(beta_term)) # (-1/1)-normalization of direction
                # beta_sol[np.abs(beta_sol)<=cg_lambda_tol] = 0 # clean small if possible
                print("beta stats:  ")
                print(np.max(beta_sol))
                print(np.min(beta_sol))
                print(np.sum(beta_sol!=0))

                # Virtual solution, so we are not checking residuals
                check_residuals = False
        else:
            beta_sol   = beta.value
            xi_sol     = mu_k_sol/2 # not really used?


        # =============================================================================
        #                     Second Stopping Criterion Condition
        # =============================================================================

        # If not a virtual solution, then check the residuals
        if check_residuals:

            # V2: Matrix of the matrix beta_k
            x_beta_z_u = beta_k.copy()

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


                
                print("Plotting bounds...")
                
                # Example data
                iterations = list(range(1, k))
                lower_bounds = lagrangian_values
                upper_bounds = master_values
                print("UB:", upper_bounds)
                print("LB:", lower_bounds)

                # Plot
                k_init = 2
                plt.plot(iterations[k_init:], upper_bounds[k_init:], label='Upper Bound', marker='s')
                plt.plot(iterations[k_init:], lower_bounds[k_init:], label='Lower Bound', marker='o')
                plt.xlabel('Iteration')
                plt.ylabel('Bound Value')
                plt.title('Bounds per Iteration')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.yscale("log")
                plt.show()


                return beta_final_sol, xi_k_sol, beta_final_sol, beta_final_sol, L.value, (t1_cg - t0_cg)/60, (t1p_cg - t0p_cg)/60, {
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

        # # Redefine unboundedness policy when close enough bounds
        # if unboundedness_policy != "v_solution":
        #     if np.abs(L_value - socp.value) <= 1e-1:
        #         unboundedness_policy = "v_solution"


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


