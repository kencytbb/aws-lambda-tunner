"""
Tests for the Intelligent Recommendation Engine.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from aws_lambda_tuner.config import TunerConfig
from aws_lambda_tuner.intelligence.recommendation_engine import IntelligentRecommendationEngine, LearningFeatures
from aws_lambda_tuner.models import MemoryTestResult, PerformanceAnalysis


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return TunerConfig(
        function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
        memory_sizes=[128, 256, 512, 1024],
        iterations=10,
        strategy="balanced",
        workload_type="on_demand",
        traffic_pattern="burst",
        cold_start_sensitivity="medium"
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
        ),
        1024: MemoryTestResult(
            memory_size=1024,
            iterations=10,
            avg_duration=1.0,
            p95_duration=1.2,
            p99_duration=1.4,
            avg_cost=0.000003,
            total_cost=0.00003,
            cold_starts=0,
            errors=0,
            raw_results=[]
        )
    }


@pytest.fixture
def sample_analysis():
    """Create sample performance analysis."""
    return PerformanceAnalysis(
        memory_results={},
        efficiency_scores={128: 0.4, 256: 0.6, 512: 0.8, 1024: 0.7},
        cost_optimal={"memory_size": 128, "cost": 0.000001},
        speed_optimal={"memory_size": 1024, "duration": 1.0},
        balanced_optimal={"memory_size": 512, "score": 0.8},
        trends={},
        insights=[]
    )


class TestIntelligentRecommendationEngine:
    """Test cases for the Intelligent Recommendation Engine."""
    
    def test_initialization(self, sample_config):
        """Test engine initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            assert engine.config == sample_config
            assert engine.data_dir.exists()
            assert engine.pattern_recognizer is not None
            assert engine.cost_predictor is not None
    
    def test_feature_extraction(self, sample_config, sample_memory_results, sample_analysis):
        """Test feature extraction for ML model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            features = engine._extract_features(sample_analysis, sample_memory_results)
            
            assert isinstance(features, LearningFeatures)
            assert features.workload_type == "on_demand"
            assert features.traffic_pattern == "burst"
            assert features.cold_start_sensitivity == "medium"
            assert features.avg_duration > 0
            assert features.memory_efficiency > 0
            assert features.cost_efficiency > 0
    
    def test_memory_efficiency_calculation(self, sample_config, sample_memory_results):
        """Test memory efficiency calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            efficiency = engine._calculate_memory_efficiency(sample_memory_results)
            
            assert isinstance(efficiency, float)
            assert efficiency > 0
    
    def test_cost_efficiency_calculation(self, sample_config, sample_memory_results):
        """Test cost efficiency calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            efficiency = engine._calculate_cost_efficiency(sample_memory_results)
            
            assert isinstance(efficiency, float)
            assert 0 <= efficiency <= 1
    
    def test_similarity_calculation(self, sample_config):
        """Test similarity score calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            features = LearningFeatures(
                workload_type="on_demand",
                traffic_pattern="burst",
                cold_start_sensitivity="medium",
                avg_duration=1.5,
                p95_duration=2.0,
                memory_efficiency=0.7,
                cost_efficiency=0.8,
                concurrency_ratio=0.1,
                error_rate=0.0,
                seasonal_variance=0.1
            )
            
            profile = {
                'workload_type': 'on_demand',
                'traffic_pattern': 'burst',
                'cold_start_sensitivity': 'medium',
                'avg_duration': 1.6,
                'memory_efficiency': 0.65
            }
            
            similarity = engine._calculate_similarity_score(features, profile)
            
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
    
    def test_confidence_score_calculation(self, sample_config):
        """Test confidence score calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            features = LearningFeatures(
                workload_type="on_demand",
                traffic_pattern="burst", 
                cold_start_sensitivity="medium",
                avg_duration=1.5,
                p95_duration=2.0,
                memory_efficiency=0.7,
                cost_efficiency=0.8,
                concurrency_ratio=0.1,
                error_rate=0.01,  # Low error rate
                seasonal_variance=0.1
            )
            
            similar_functions = ["func1", "func2", "func3"]
            
            confidence = engine._calculate_confidence_score(features, similar_functions)
            
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
    
    def test_risk_assessment(self, sample_config):
        """Test optimization risk assessment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            features = LearningFeatures(
                workload_type="on_demand",
                traffic_pattern="burst",
                cold_start_sensitivity="medium", 
                avg_duration=1.5,
                p95_duration=2.0,
                memory_efficiency=0.7,
                cost_efficiency=0.8,
                concurrency_ratio=0.1,
                error_rate=0.02,  # Some errors
                seasonal_variance=0.1
            )
            
            from aws_lambda_tuner.models import Recommendation
            recommendation = Recommendation(
                strategy="balanced",
                current_memory_size=256,
                optimal_memory_size=512,  # 100% increase
                should_optimize=True,
                cost_change_percent=10.0,
                duration_change_percent=-20.0
            )
            
            risk_assessment = engine._assess_optimization_risks(features, recommendation)
            
            assert isinstance(risk_assessment, dict)
            assert 'level' in risk_assessment
            assert 'factors' in risk_assessment
            assert 'mitigation' in risk_assessment
    
    @patch('aws_lambda_tuner.intelligence.recommendation_engine.datetime')
    def test_learning_data_storage(self, mock_datetime, sample_config):
        """Test storage of learning data."""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            features = LearningFeatures(
                workload_type="on_demand",
                traffic_pattern="burst",
                cold_start_sensitivity="medium",
                avg_duration=1.5,
                p95_duration=2.0,
                memory_efficiency=0.7,
                cost_efficiency=0.8,
                concurrency_ratio=0.1,
                error_rate=0.0,
                seasonal_variance=0.1
            )
            
            from aws_lambda_tuner.intelligence.recommendation_engine import MLRecommendation
            from aws_lambda_tuner.models import Recommendation
            
            base_rec = Recommendation(
                strategy="balanced",
                current_memory_size=256,
                optimal_memory_size=512,
                should_optimize=True
            )
            
            ml_rec = MLRecommendation(
                base_recommendation=base_rec,
                confidence_score=0.8,
                pattern_match_score=0.7,
                similar_functions=[],
                predicted_performance={},
                risk_assessment={},
                optimization_timeline={}
            )
            
            # This should not raise an exception
            engine._store_learning_data(features, ml_rec)
            
            # Check that learning file was created
            learning_files = list(engine.data_dir.glob("learning_data_*.jsonl"))
            assert len(learning_files) > 0
    
    def test_generate_intelligent_recommendation(self, sample_config, sample_memory_results, sample_analysis):
        """Test generation of intelligent recommendations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = IntelligentRecommendationEngine(sample_config, temp_dir)
            
            # Mock cost predictor to avoid complex setup
            engine.cost_predictor.predict_costs_for_memory = Mock(return_value={
                'cost_per_invocation': 0.000002,
                'monthly_cost': 60.0
            })
            
            recommendation = engine.generate_intelligent_recommendation(sample_analysis, sample_memory_results)
            
            assert recommendation is not None
            assert hasattr(recommendation, 'base_recommendation')
            assert hasattr(recommendation, 'confidence_score')
            assert hasattr(recommendation, 'pattern_match_score')
            assert hasattr(recommendation, 'risk_assessment')
            
            # Check that confidence score is reasonable
            assert 0 <= recommendation.confidence_score <= 1
            
            # Check that base recommendation is valid
            base_rec = recommendation.base_recommendation
            assert base_rec.optimal_memory_size in sample_memory_results.keys()
            assert isinstance(base_rec.should_optimize, bool)


class TestLearningFeatures:
    """Test cases for learning features data structure."""
    
    def test_learning_features_creation(self):
        """Test creation of learning features."""
        features = LearningFeatures(
            workload_type="on_demand",
            traffic_pattern="burst",
            cold_start_sensitivity="high",
            avg_duration=1.5,
            p95_duration=2.0,
            memory_efficiency=0.7,
            cost_efficiency=0.8,
            concurrency_ratio=0.1,
            error_rate=0.02,
            seasonal_variance=0.15
        )
        
        assert features.workload_type == "on_demand"
        assert features.traffic_pattern == "burst"
        assert features.cold_start_sensitivity == "high"
        assert features.avg_duration == 1.5
        assert features.p95_duration == 2.0
        assert features.memory_efficiency == 0.7
        assert features.cost_efficiency == 0.8
        assert features.concurrency_ratio == 0.1
        assert features.error_rate == 0.02
        assert features.seasonal_variance == 0.15


# Integration test
def test_full_recommendation_workflow(sample_config, sample_memory_results, sample_analysis):
    """Test the complete recommendation workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        engine = IntelligentRecommendationEngine(sample_config, temp_dir)
        
        # Mock dependencies to focus on workflow
        engine.cost_predictor.predict_costs_for_memory = Mock(return_value={
            'cost_per_invocation': 0.000002,
            'monthly_cost': 60.0
        })
        
        # Generate recommendation
        recommendation = engine.generate_intelligent_recommendation(sample_analysis, sample_memory_results)
        
        # Verify complete recommendation structure
        assert recommendation is not None
        assert hasattr(recommendation, 'base_recommendation')
        assert hasattr(recommendation, 'confidence_score')
        assert hasattr(recommendation, 'pattern_match_score')
        assert hasattr(recommendation, 'similar_functions')
        assert hasattr(recommendation, 'predicted_performance')
        assert hasattr(recommendation, 'risk_assessment')
        assert hasattr(recommendation, 'optimization_timeline')
        
        # Verify recommendation makes sense
        base_rec = recommendation.base_recommendation
        assert base_rec.optimal_memory_size in [128, 256, 512, 1024]
        assert 0 <= recommendation.confidence_score <= 1
        assert 0 <= recommendation.pattern_match_score <= 1
        
        # Verify risk assessment structure
        risk = recommendation.risk_assessment
        assert 'level' in risk
        assert risk['level'] in ['low', 'medium', 'high']
        
        # Verify timeline structure
        timeline = recommendation.optimization_timeline
        assert 'immediate' in timeline
        assert 'short_term' in timeline
        assert 'long_term' in timeline