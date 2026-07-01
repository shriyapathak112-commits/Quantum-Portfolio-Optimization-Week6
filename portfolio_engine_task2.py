import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

def run_portfolio_engine(csv_path, budget, risk_preference, asset_constraints=None):
    """
    Core Portfolio Optimization Engine matching all Task 2 requirements.
    """
    print(f"\n--- Loading Data from: {csv_path} ---")
    
    # 1. Inputs processing: Parse Portfolio CSV
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    asset_names = df.columns.tolist()
    num_assets = len(asset_names)
    
    # Compute returns, expected return vector, and covariance risk matrix
    returns = df.pct_change().dropna()
    expected_returns = returns.mean().to_numpy()
    covariance_matrix = returns.cov().to_numpy()
    
    # 2. Build QUBO with User Inputs & Constraints
    qp = QuadraticProgram("Dynamic_Portfolio_Engine")
    for asset in asset_names:
        qp.binary_var(name=asset)
        
    # Set Objective using user risk preference
    # Minimizing (-returns + risk) maximizes the risk-adjusted return profile
    qp.minimize(linear=-expected_returns, quadratic=risk_preference * covariance_matrix)
    
    # Enforce Budget Constraint (pick exactly 'budget' number of assets)
    budget_dict = {asset: 1 for asset in asset_names}
    qp.linear_constraint(linear=budget_dict, sense="==", rhs=budget, name="Budget")
    
    # Enforce Asset Constraints (Example: Force a specific asset to be included if requested)
    if asset_constraints and "must_include" in asset_constraints:
        target_asset = asset_constraints["must_include"]
        if target_asset in asset_names:
            qp.linear_constraint(linear={target_asset: 1}, sense="==", rhs=1, name="Force_Include")
            print(f"Applied Constraint: Forcing inclusion of {target_asset}")

    # 3. Optimize Portfolio via Hybrid Engine Solver
    exact_mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(exact_mes)
    result = optimizer.solve(qp)
    
    # 4. Process Outputs: Map binary array selections to fractional Allocation Weights
    selection_vector = result.x
    selected_indices = [i for i, val in enumerate(selection_vector) if val == 1]
    
    # Generate Equal Allocation Weights across chosen portfolio assets
    allocation_weights = {}
    for i, asset in enumerate(asset_names):
        if selection_vector[i] == 1:
            allocation_weights[asset] = f"{(1.0 / len(selected_indices)) * 100:.2f}%"
        else:
            allocation_weights[asset] = "0.00%"
            
    # Calculate Final Risk Metrics
    portfolio_return = np.dot(selection_vector, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(selection_vector, np.dot(covariance_matrix, selection_vector)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # 5. Display Deliverable Engine Report Output
    print("\n================ ENGINE OPTIMIZATION OUTPUT ================")
    print(f"Optimized Portfolio Selection:  {selection_vector}")
    print(f"Calculated Allocation Weights:  {allocation_weights}")
    print(f"Expected Allocation Return:     {portfolio_return:.6f}")
    print(f"Portfolio Volatility Risk:      {portfolio_volatility:.6f}")
    print(f"Calculated Sharpe Ratio:        {sharpe_ratio:.4f}")
    print("=============================================================")

if __name__ == "__main__":
    # Define User Configurations dynamically here:
    USER_CSV = "portfolio_data.csv"
    USER_BUDGET = 2                 # Total number of assets to pick
    USER_RISK_PREF = 0.5            # Higher means more risk-averse
    
    # Optional asset constraint rules (e.g. force picking a specific stock ticker)
    USER_CONSTRAINTS = {"must_include": "AAPL"} 
    
    # Run the engine
    run_portfolio_engine(
        csv_path=USER_CSV, 
        budget=USER_BUDGET, 
        risk_preference=USER_RISK_PREF, 
        asset_constraints=USER_CONSTRAINTS
    )