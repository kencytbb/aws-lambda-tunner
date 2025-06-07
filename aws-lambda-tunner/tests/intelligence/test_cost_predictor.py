"""
Tests for the Cost Predictor component.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from aws_lambda_tuner.config import TunerConfig
from aws_lambda_tuner.intelligence.cost_predictor import CostPredictor, CostPrediction, WorkloadCostModel
from aws_lambda_tuner.models import MemoryTestResult


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
        memory_sizes=[128, 256, 512, 1024],
        iterations=10,
        strategy="balanced",
        workload_type="on_demand",
        cost_per_gb_second=0.0000166667,
        cost_per_request=0.0000002
    )


@pytest.fixture
def sample_memory_results():
    """Create sample memory test results."""
    return {
        128: MemoryTestResult(
            memory_size=128,
            iterations=10,
            avg_duration=2.5,
            p95_duration=3.0,
            p99_duration=3.5,
            avg_cost=0.000001,
            total_cost=0.00001,
            cold_starts=3,
            errors=0,
            raw_results=[]
        ),
        256: MemoryTestResult(
            memory_size=256,
            iterations=10,
            avg_duration=1.8,
            p95_duration=2.2,
            p99_duration=2.5,
            avg_cost=0.0000015,
            total_cost=0.000015,
            cold_starts=2,
            errors=0,
            raw_results=[]
        ),
        512: MemoryTestResult(
            memory_size=512,
            iterations=10,
            avg_duration=1.2,
            p95_duration=1.5,
            p99_duration=1.8,
            avg_cost=0.000002,
            total_cost=0.00002,
            cold_starts=1,
            errors=0,
            raw_results=[]
        )
    }


class TestCostPredictor:
    """Test cases for the Cost Predictor."""
    
    def test_initialization(self, sample_config):
        """Test cost predictor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            assert predictor.config == sample_config
            assert predictor.data_dir.exists()
            assert predictor.cost_per_gb_second == sample_config.cost_per_gb_second
            assert predictor.cost_per_request == sample_config.cost_per_request
            assert len(predictor.workload_models) > 0
    
    def test_workload_models_initialization(self, sample_config):
        """Test workload models are properly initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            models = predictor.workload_models
            
            assert 'on_demand' in models
            assert 'scheduled' in models
            assert 'continuous' in models
            
            # Check model structure
            on_demand_model = models['on_demand']
            assert isinstance(on_demand_model, WorkloadCostModel)
            assert on_demand_model.base_cost_factor > 0
            assert on_demand_model.duration_sensitivity > 0
            assert len(on_demand_model.memory_efficiency_curve) > 0
    
    def test_get_workload_model(self, sample_config):
        """Test getting the appropriate workload model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            model = predictor._get_workload_model()
            
            # Should return on_demand model based on config
            assert model.workload_type.value == "on_demand"
    
    def test_base_cost_calculation(self, sample_config):
        """Test base AWS Lambda cost calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            memory_size = 512  # MB
            avg_duration = 1.5  # seconds
            monthly_invocations = 100000
            
            base_costs = predictor._calculate_base_costs(
                memory_size, avg_duration, monthly_invocations
            )
            
            assert 'cost_per_invocation' in base_costs
            assert 'total_cost' in base_costs
            assert 'breakdown' in base_costs
            
            assert base_costs['cost_per_invocation'] > 0
            assert base_costs['total_cost'] > 0
            assert 'compute' in base_costs['breakdown']
            assert 'requests' in base_costs['breakdown']
    
    def test_workload_adjustments(self, sample_config):
        """Test workload-specific cost adjustments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            base_costs = {
                'cost_per_invocation': 0.000002,
                'total_cost': 200.0,
                'breakdown': {'compute': 180.0, 'requests': 20.0}
            }
            
            workload_model = predictor._get_workload_model()
            workload_characteristics = {
                'cold_start_ratio': 0.1,
                'concurrency_ratio': 0.3,
                'error_ratio': 0.01
            }
            
            adjusted_costs = predictor._apply_workload_adjustments(
                base_costs, workload_model, workload_characteristics
            )
            
            assert 'cost_per_invocation' in adjusted_costs
            assert 'total_cost' in adjusted_costs
            assert 'breakdown' in adjusted_costs
            
            # Adjustments should change the costs
            assert adjusted_costs['total_cost'] != base_costs['total_cost']
    
    def test_confidence_interval_calculation(self, sample_config):
        """Test confidence interval calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            predicted_cost = 100.0
            memory_size = 512
            avg_duration = 1.5
            
            confidence_interval = predictor._calculate_confidence_interval(
                predicted_cost, memory_size, avg_duration
            )
            
            assert isinstance(confidence_interval, tuple)
            assert len(confidence_interval) == 2
            
            lower_bound, upper_bound = confidence_interval
            assert lower_bound < predicted_cost < upper_bound
            assert lower_bound > 0
            assert upper_bound > predicted_cost
    
    def test_prediction_confidence_calculation(self, sample_config):
        """Test prediction confidence calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            # Test with common memory size and typical values
            confidence = predictor._calculate_prediction_confidence(
                memory_size=512,  # Common size
                avg_duration=1.5,  # Typical duration
                monthly_invocations=10000  # Sufficient data
            )
            
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            
            # Test with uncommon memory size
            confidence_uncommon = predictor._calculate_prediction_confidence(
                memory_size=333,  # Uncommon size
                avg_duration=0.05,  # Very short duration
                monthly_invocations=50  # Low data
            )
            
            assert confidence_uncommon < confidence  # Should have lower confidence
    
    def test_savings_potential_calculation(self, sample_config):
        """Test savings potential calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            # Test with more efficient configuration
            savings = predictor._calculate_savings_potential(
                memory_size=512,
                predicted_cost=80.0  # Lower than baseline
            )
            
            assert isinstance(savings, float)
            assert savings >= 0  # Should show savings
    
    def test_predict_costs_for_memory(self, sample_config):
        """Test cost prediction for specific memory configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            prediction = predictor.predict_costs_for_memory(
                memory_size=512,
                avg_duration=1.5,
                monthly_invocations=10000,
                workload_characteristics={'cold_start_ratio': 0.1}
            )
            
            assert isinstance(prediction, CostPrediction)
            assert prediction.memory_size == 512
            assert prediction.predicted_cost_per_invocation > 0
            assert prediction.predicted_monthly_cost > 0
            assert isinstance(prediction.confidence_interval, tuple)
            assert isinstance(prediction.cost_breakdown, dict)
            assert prediction.savings_potential >= 0
            assert 0 <= prediction.prediction_confidence <= 1
    
    def test_predict_costs_across_memories(self, sample_config, sample_memory_results):
        """Test cost prediction across multiple memory configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            memory_sizes = [128, 256, 512, 1024]
            monthly_invocations = 10000
            
            predictions = predictor.predict_costs_across_memories(
                memory_sizes, sample_memory_results, monthly_invocations
            )
            
            assert isinstance(predictions, dict)
            assert len(predictions) == len(memory_sizes)
            
            for memory_size in memory_sizes:
                assert memory_size in predictions
                prediction = predictions[memory_size]
                assert isinstance(prediction, CostPrediction)
                assert prediction.memory_size == memory_size
    
    def test_find_cost_optimal_configuration(self, sample_config):
        """Test finding cost-optimal configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            # Create mock predictions
            cost_predictions = {
                128: CostPrediction(
                    memory_size=128,
                    predicted_cost_per_invocation=0.000001,
                    predicted_monthly_cost=100.0,
                    confidence_interval=(90.0, 110.0),
                    cost_breakdown={'compute': 80.0, 'requests': 20.0},
                    savings_potential=0.0,
                    prediction_confidence=0.8
                ),
                256: CostPrediction(
                    memory_size=256,
                    predicted_cost_per_invocation=0.0000015,
                    predicted_monthly_cost=150.0,
                    confidence_interval=(135.0, 165.0),
                    cost_breakdown={'compute': 130.0, 'requests': 20.0},
                    savings_potential=5.0,
                    prediction_confidence=0.8
                ),
                512: CostPrediction(
                    memory_size=512,
                    predicted_cost_per_invocation=0.000002,
                    predicted_monthly_cost=80.0,  # Most cost-effective
                    confidence_interval=(72.0, 88.0),
                    cost_breakdown={'compute': 60.0, 'requests': 20.0},
                    savings_potential=20.0,
                    prediction_confidence=0.9
                )
            }
            
            optimal_memory, optimal_prediction = predictor.find_cost_optimal_configuration(
                cost_predictions
            )
            
            assert optimal_memory == 512  # Should pick the lowest cost option
            assert optimal_prediction.memory_size == 512
            assert optimal_prediction.predicted_monthly_cost == 80.0
    
    def test_analyze_cost_trends(self, sample_config):
        """Test cost trend analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            # Create mock predictions with clear trend
            cost_predictions = {
                128: CostPrediction(
                    memory_size=128, predicted_monthly_cost=120.0,
                    predicted_cost_per_invocation=0.000001, confidence_interval=(0, 0),
                    cost_breakdown={}, savings_potential=0.0, prediction_confidence=0.8
                ),
                256: CostPrediction(
                    memory_size=256, predicted_monthly_cost=100.0,
                    predicted_cost_per_invocation=0.000001, confidence_interval=(0, 0),
                    cost_breakdown={}, savings_potential=0.0, prediction_confidence=0.8
                ),
                512: CostPrediction(
                    memory_size=512, predicted_monthly_cost=90.0,
                    predicted_cost_per_invocation=0.000001, confidence_interval=(0, 0),
                    cost_breakdown={}, savings_potential=0.0, prediction_confidence=0.8
                ),
                1024: CostPrediction(
                    memory_size=1024, predicted_monthly_cost=110.0,
                    predicted_cost_per_invocation=0.000001, confidence_interval=(0, 0),
                    cost_breakdown={}, savings_potential=0.0, prediction_confidence=0.8
                )
            }
            
            trends = predictor.analyze_cost_trends(cost_predictions)
            
            assert isinstance(trends, dict)
            assert 'trend' in trends
            assert 'optimal_range' in trends
            assert 'recommendations' in trends
            
            # Should identify 512MB as optimal
            assert trends['optimal_range']['min_memory'] == 512
            assert trends['optimal_range']['min_cost'] == 90.0
    
    def test_duration_estimation(self, sample_config, sample_memory_results):
        """Test duration estimation for missing memory configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            # Estimate duration for 384MB (between 256MB and 512MB)
            estimated_duration = predictor._estimate_duration_for_memory(
                384, sample_memory_results
            )
            
            assert isinstance(estimated_duration, float)
            assert estimated_duration > 0
            
            # Should be between 256MB and 512MB durations
            duration_256 = sample_memory_results[256].avg_duration
            duration_512 = sample_memory_results[512].avg_duration
            
            assert min(duration_256, duration_512) <= estimated_duration <= max(duration_256, duration_512)
    
    def test_workload_characteristics_extraction(self, sample_config):
        """Test extraction of workload characteristics from test results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            result = MemoryTestResult(
                memory_size=512,
                iterations=10,
                avg_duration=1.5,
                p95_duration=2.0,
                p99_duration=2.5,
                avg_cost=0.000002,
                total_cost=0.00002,
                cold_starts=2,
                errors=1,
                raw_results=[]
            )
            
            characteristics = predictor._extract_workload_characteristics(result)
            
            assert isinstance(characteristics, dict)
            assert 'cold_start_ratio' in characteristics
            assert 'error_ratio' in characteristics
            assert 'duration_variance' in characteristics
            assert 'concurrency_ratio' in characteristics
            
            # Check calculated values
            assert characteristics['cold_start_ratio'] == 0.2  # 2/10
            assert characteristics['error_ratio'] == 0.1  # 1/10
    
    def test_cost_trend_recommendations(self, sample_config):
        """Test generation of cost trend recommendations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            memory_sizes = [128, 256, 512, 1024]
            costs = [120.0, 100.0, 90.0, 110.0]  # Sweet spot at 512MB
            sweet_spots = [512]
            
            recommendations = predictor._generate_cost_trend_recommendations(
                memory_sizes, costs, sweet_spots
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            # Should mention the sweet spot
            sweet_spot_mentioned = any("512" in rec for rec in recommendations)
            assert sweet_spot_mentioned
    
    @patch('aws_lambda_tuner.intelligence.cost_predictor.datetime')
    def test_prediction_storage(self, mock_datetime, sample_config):
        """Test storage of cost predictions for learning."""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value.strftime.return_value = "202401"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            prediction = CostPrediction(
                memory_size=512,
                predicted_cost_per_invocation=0.000002,
                predicted_monthly_cost=100.0,
                confidence_interval=(90.0, 110.0),
                cost_breakdown={'compute': 80.0, 'requests': 20.0},
                savings_potential=10.0,
                prediction_confidence=0.8
            )
            
            characteristics = {'cold_start_ratio': 0.1}
            
            # Should not raise an exception
            predictor._store_prediction(prediction, characteristics)
            
            # Check that prediction file was created
            prediction_files = list(predictor.data_dir.glob("cost_predictions_*.jsonl"))
            assert len(prediction_files) > 0
    
    def test_basic_cost_calculation_fallback(self, sample_config):
        """Test basic cost calculation as fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CostPredictor(sample_config, temp_dir)
            
            basic_cost = predictor._calculate_basic_cost(
                memory_size=512,
                avg_duration=1.5,
                monthly_invocations=10000
            )
            
            assert isinstance(basic_cost, float)
            assert basic_cost > 0
            
            # Should be sum of compute and request costs
            memory_gb = 512 / 1024.0
            expected_compute = memory_gb * 1.5 * predictor.cost_per_gb_second * 10000
            expected_request = predictor.cost_per_request * 10000
            expected_total = expected_compute + expected_request
            
            assert abs(basic_cost - expected_total) < 0.000001  # Close to expected