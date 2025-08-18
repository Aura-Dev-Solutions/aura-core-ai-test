"""
A/B Testing system for model deployment and experimentation.
"""

import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class ExperimentVariant:
    name: str
    weight: float  # Traffic allocation (0.0 to 1.0)
    model_config: Dict[str, Any]


@dataclass
class ExperimentMetrics:
    variant_name: str
    total_requests: int = 0
    successful_requests: int = 0
    average_latency: float = 0.0
    accuracy_score: float = 0.0


class ABTestingManager(LoggerMixin):
    """Manager for A/B testing experiments on AI models."""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, ExperimentMetrics]] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: List[ExperimentVariant]
    ) -> bool:
        """Create a new A/B testing experiment."""
        try:
            total_weight = sum(v.weight for v in variants)
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
            
            self.experiments[experiment_id] = {
                "name": name,
                "variants": {v.name: v for v in variants},
                "status": ExperimentStatus.DRAFT,
                "created_at": datetime.utcnow()
            }
            
            self.metrics[experiment_id] = {
                v.name: ExperimentMetrics(variant_name=v.name)
                for v in variants
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment {experiment_id}: {str(e)}")
            return False
    
    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign a user to a variant using consistent hashing."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        if experiment["status"] != ExperimentStatus.RUNNING:
            return None
        
        # Check existing assignment
        if experiment_id in self.user_assignments:
            if user_id in self.user_assignments[experiment_id]:
                return self.user_assignments[experiment_id][user_id]
        
        # Use consistent hashing
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Assign based on weights
        cumulative_weight = 0.0
        variants = experiment["variants"]
        
        for variant_name, variant in variants.items():
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                if experiment_id not in self.user_assignments:
                    self.user_assignments[experiment_id] = {}
                self.user_assignments[experiment_id][user_id] = variant_name
                return variant_name
        
        return list(variants.keys())[0]
    
    def record_metrics(self, experiment_id: str, variant_name: str, latency: float, success: bool, accuracy: Optional[float] = None):
        """Record metrics for an experiment variant."""
        if experiment_id not in self.metrics or variant_name not in self.metrics[experiment_id]:
            return
        
        metrics = self.metrics[experiment_id][variant_name]
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
        
        # Update running average for latency
        if metrics.total_requests == 1:
            metrics.average_latency = latency
        else:
            metrics.average_latency = (
                (metrics.average_latency * (metrics.total_requests - 1) + latency) /
                metrics.total_requests
            )
        
        # Update accuracy if provided
        if accuracy is not None:
            if metrics.total_requests == 1:
                metrics.accuracy_score = accuracy
            else:
                metrics.accuracy_score = (
                    (metrics.accuracy_score * (metrics.total_requests - 1) + accuracy) /
                    metrics.total_requests
                )


# Global instance
ab_testing_manager = ABTestingManager()
