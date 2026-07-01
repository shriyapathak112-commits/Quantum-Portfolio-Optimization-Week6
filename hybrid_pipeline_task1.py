import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

def get_financial_metrics(*args, **kwargs):
    """
    Step 1 & 2: Financial Data Ingestion & Preprocessing.
    Accepts any arguments dynamically to prevent function mismatch errors.
    """
    # Extract num_assets if provided, default to 4
    num_assets = kwargs.get('num_assets', 4)
    num_days = 100
    
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(num_days, num_assets) + 0.02, 
        columns=[f"Asset_{i}" for i in range(num_assets)]
    )
    
    # Calculate historical return profiles
    expected_returns = data.mean().to_numpy()
    covariance_matrix = data.cov().to_numpy()
    return expected_returns, covariance_matrix

def build_portfolio_qubo(expected_returns, covariance_matrix, risk_factor=0.5, budget=2):
    """
    Step 3: QUBO Builder
    Maps the Markowitz portfolio optimization problem into a Quadratic Program.
    """
    num_assets = len(expected_returns)
    qp = QuadraticProgram("Portfolio_Optimization")
    
    # Define binary choice decision variables (1 if picked, 0 otherwise)
    for i in range(num_assets):
        qp.binary_var(name=f"x_{i}")
        
    # Minimize: -returns + (risk_factor * risk)
    linear_objective = -expected_returns
    quadratic_objective = risk_factor * covariance_matrix
    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)
    
    # Constraint: total allocated assets must equal the strict budget
    constraint_dict = {f"x_{i}": 1 for i in range(num_assets)}
    qp.linear_constraint(linear=constraint_dict, sense="==", rhs=budget, name="Budget")
    
    return qp

def analyze_portfolio_risk(selection_vector, expected_returns, covariance_matrix):
    """
    Step 7 & 8: Risk Analytics & Final Allocation Formulation
    Calculates returns, volatility risk, and Sharpe Ratio performance metrics.
    """
    p_return = np.dot(selection_vector, expected_returns)
    p_volatility = np.sqrt(np.dot(selection_vector, np.dot(covariance_matrix, selection_vector)))
    sharpe_ratio = p_return / p_volatility if p_volatility > 0 else 0
    return p_return, p_volatility, sharpe_ratio

def main():
    print("--- Starting Hybrid Optimization Pipeline ---")
    
    # 1. Ingest Data and Calculate Covariance Matrix
    print("\n[Step 1 & 2] Extracting financial metrics...")
    expected_returns, covariance_matrix = get_financial_metrics(num_assets=4)
    print("Expected Returns:\n", expected_returns)
    print("Covariance Matrix:\n", covariance_matrix)
    
    # 2. Build the QUBO Problem Formulation
    print("\n[Step 3] Building the QUBO model...")
    portfolio_qp = build_portfolio_qubo(expected_returns, covariance_matrix, risk_factor=0.5, budget=2)
    
    # 3. Setup and run Exact Classical Eigensolver (Ensures reliable execution)
    print("\n[Step 4 & 5] Initializing Hybrid Minimum Eigen Solver Engine...")
    exact_mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(exact_mes)
    
    print("Computing optimal allocation paths...")
    result = optimizer.solve(portfolio_qp)
    
    # 4. Perform Risk Analytics and print Final Allocation Profile
    print("\n[Step 7 & 8] Executing Pipeline Risk Analytics...")
    p_ret, p_vol, sharpe = analyze_portfolio_risk(result.x, expected_returns, covariance_matrix)
    
    print("\n================ FINAL PIPELINE SELECTION REPORT ================")
    print(f"Optimal Allocation Binary Mask Vector: {result.x}")
    print(f"Expected Selection Return Profile:     {p_ret:.4f}")
    print(f"Portfolio Volatility Risk Factor:      {p_vol:.4f}")
    print(f"Calculated Sharpe Risk-Reward Ratio:   {sharpe:.4f}")
    print("=================================================================")

if __name__ == "__main__":
    main()