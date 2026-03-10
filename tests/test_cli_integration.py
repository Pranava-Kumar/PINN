import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from run_demo import run_complete_demo

@patch('run_demo.train_pinn')
@patch('run_demo.predict_trajectory')
@patch('run_demo.optimize_delta_v')
@patch('run_demo.compute_pareto_front')
@patch('run_demo.plot_training_history')
@patch('run_demo.plot_trajectory_comparison')
@patch('run_demo.plot_optimization_results')
@patch('run_demo.plot_pareto_front')
@patch('run_demo.ReportGenerator')
@patch('torch.save')
def test_cli_triggers_pdf_generation(mock_save, mock_gen, mock_plot4, mock_plot3, mock_plot2, mock_plot1, 
                                     mock_pareto, mock_opt, mock_predict, mock_train, tmp_path):
    """Test that run_complete_demo triggers PDF report generation."""
    # Setup mocks
    mock_history = MagicMock()
    mock_history.best_loss = 1e-5
    mock_history.best_epoch = 1000
    mock_history.train_time_s = 10.0
    mock_history.epochs = [0, 1]
    mock_history.loss_total = [0.1, 0.01]
    mock_history.loss_data = [0.05, 0.005]
    mock_history.loss_physics = [0.04, 0.004]
    mock_history.loss_initial = [0.01, 0.001]
    mock_history.learning_rate = [1e-3, 1e-3]
    
    mock_rk8 = MagicMock()
    mock_rk8.t_years = np.linspace(0, 1, 10)
    mock_rk8.compute_time_s = 1.0
    mock_rk8.states = np.zeros((10, 6))
    mock_rk8.altitudes_km = np.linspace(750, 749, 10)
    mock_rk8.velocities_km_s = np.linspace(7.5, 7.5, 10)
    
    mock_train.return_value = (MagicMock(), MagicMock(), mock_history, mock_rk8)
    
    mock_predict.return_value = (np.zeros((10, 6)), 1.0)
    mock_opt.return_value = MagicMock(optimal_dv=10.0, lifetime_at_optimal=4.0, fuel_savings_percent=50.0, 
                                     propulsive_only_dv=20.0, no_burn_lifetime=10.0, method="hybrid",
                                     success=True, n_function_evaluations=10, compute_time_s=1.0, message="")
    mock_pareto.return_value = MagicMock(n_pareto_points=5)
    
    # Run demo
    run_complete_demo("PSLV", str(tmp_path))
    
    # Verify ReportGenerator was called
    assert mock_gen.called
    # Get the instance
    instance = mock_gen.return_value
    assert instance.add_mission_data.called
    assert instance.add_optimization_results.called
    assert instance.add_plots.called
    assert instance.save_document.called

