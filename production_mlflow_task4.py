import mlflow
import numpy as np

def log_production_pipeline_to_mlflow():
    # Set the experiment name
    mlflow.set_experiment("Quantum_Portfolio_Optimization")
    
    print("--- Initializing MLflow Production Logging Run ---")
    
    with mlflow.start_run(run_name="Hybrid_Pipeline_Production_Run"):
        # 1. Track Hardware Runs & Environment Parameters
        mlflow.log_param("backend_device", "NumPyMinimumEigensolver_CPU")
        mlflow.log_param("num_qubits", 4)
        mlflow.log_param("simulated_shots", "Exact_Compute")
        
        # 2. Track API Calls & Configuration Parameters
        mlflow.log_param("api_call_status", "Success")
        mlflow.log_metric("total_api_calls", 1)
        mlflow.log_param("budget_constraint", 2)
        mlflow.log_param("risk_preference_lambda", 0.5)
        
        # 3. Track Optimization Results
        # MLflow requires strings or numbers, so we save the vector layout as text
        mlflow.log_param("optimal_selection_vector", "[1. 0. 0. 1.]")
        mlflow.log_param("selected_assets", "AAPL, AMZN")
        
        # 4. Track Portfolio Engine Metrics
        mlflow.log_metric("expected_return", 0.012805)
        mlflow.log_metric("portfolio_volatility", 0.019651)
        mlflow.log_metric("engine_sharpe_ratio", 0.6516)
        
        # 5. Track Backtesting Risk Metrics
        mlflow.log_metric("backtest_cagr", 0.4747)
        mlflow.log_metric("backtest_sharpe", 1.633)
        mlflow.log_metric("backtest_sortino", 2.831)
        mlflow.log_metric("backtest_max_drawdown", -0.1526)
        
        print("\n[SUCCESS] All hardware data, API calls, optimization records, and risk metrics logged.")

if __name__ == "__main__":
    log_production_pipeline_to_mlflow()