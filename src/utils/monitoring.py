"""
Advanced monitoring and alerting system for the document analyzer.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict

from src.core.logging import LoggerMixin


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    severity: AlertSeverity
    duration: int  # seconds the condition must persist
    callback: Optional[Callable] = None


@dataclass
class Alert:
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    message: str = ""


class MetricCollector:
    """Collects and stores metrics with time-series data."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonically increasing)."""
        key = self._get_metric_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = {
                "type": MetricType.COUNTER,
                "value": 0.0,
                "labels": labels or {}
            }
        
        self.metrics[key]["value"] += value
        self.time_series[key].append((time.time(), self.metrics[key]["value"]))
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (can go up or down)."""
        key = self._get_metric_key(name, labels)
        
        self.metrics[key] = {
            "type": MetricType.GAUGE,
            "value": value,
            "labels": labels or {}
        }
        
        self.time_series[key].append((time.time(), value))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (for measuring distributions)."""
        key = self._get_metric_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = {
                "type": MetricType.HISTOGRAM,
                "values": [],
                "labels": labels or {}
            }
        
        self.metrics[key]["values"].append(value)
        # Keep only last 1000 values for memory efficiency
        if len(self.metrics[key]["values"]) > 1000:
            self.metrics[key]["values"] = self.metrics[key]["values"][-1000:]
        
        self.time_series[key].append((time.time(), value))
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value of a metric."""
        key = self._get_metric_key(name, labels)
        
        if key not in self.metrics:
            return None
        
        metric = self.metrics[key]
        
        if metric["type"] in [MetricType.COUNTER, MetricType.GAUGE]:
            return metric["value"]
        elif metric["type"] == MetricType.HISTOGRAM:
            values = metric["values"]
            return statistics.mean(values) if values else 0.0
        
        return None
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
        """Get statistics for a histogram metric."""
        key = self._get_metric_key(name, labels)
        
        if key not in self.metrics or self.metrics[key]["type"] != MetricType.HISTOGRAM:
            return None
        
        values = self.metrics[key]["values"]
        if not values:
            return None
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


class AlertManager(LoggerMixin):
    """Manages alert rules and triggers alerts based on metrics."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.rule_states: Dict[str, Dict[str, Any]] = {}
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to be monitored."""
        self.alert_rules[rule.name] = rule
        self.rule_states[rule.name] = {
            "condition_start": None,
            "last_check": None
        }
        
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if conditions are met."""
        current_time = datetime.utcnow()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                self._check_single_rule(rule, current_time)
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {str(e)}")
    
    def _check_single_rule(self, rule: AlertRule, current_time: datetime):
        """Check a single alert rule."""
        metric_value = self.metric_collector.get_metric_value(rule.metric_name)
        
        if metric_value is None:
            return
        
        condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
        rule_state = self.rule_states[rule.name]
        
        if condition_met:
            if rule_state["condition_start"] is None:
                rule_state["condition_start"] = current_time
            
            # Check if condition has persisted long enough
            duration = (current_time - rule_state["condition_start"]).total_seconds()
            
            if duration >= rule.duration and rule.name not in self.active_alerts:
                # Trigger alert
                alert = Alert(
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    triggered_at=current_time,
                    message=f"Metric {rule.metric_name} is {metric_value}, threshold: {rule.threshold}"
                )
                
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(
                    f"Alert triggered: {rule.name}",
                    metric=rule.metric_name,
                    value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity.value
                )
                
                # Execute callback if provided
                if rule.callback:
                    try:
                        rule.callback(alert)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed for {rule.name}: {str(e)}")
        
        else:
            # Condition not met, reset state
            rule_state["condition_start"] = None
            
            # Resolve alert if it was active
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                alert.resolved_at = current_time
                del self.active_alerts[rule.name]
                
                self.logger.info(f"Alert resolved: {rule.name}")
        
        rule_state["last_check"] = current_time
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if a condition is met."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        else:
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())


class MonitoringSystem(LoggerMixin):
    """Main monitoring system that coordinates metrics collection and alerting."""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager(self.metric_collector)
        self.last_alert_check = time.time()
        self.alert_check_interval = 30  # seconds
    
    def record_request_metrics(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        labels = {"endpoint": endpoint, "method": method, "status": str(status_code)}
        
        self.metric_collector.record_counter("http_requests_total", 1.0, labels)
        self.metric_collector.record_histogram("http_request_duration_seconds", duration, labels)
        
        if status_code >= 400:
            self.metric_collector.record_counter("http_errors_total", 1.0, labels)
    
    def record_ai_metrics(self, model_name: str, operation: str, duration: float, success: bool, accuracy: Optional[float] = None):
        """Record AI model performance metrics."""
        labels = {"model": model_name, "operation": operation}
        
        self.metric_collector.record_counter("ai_operations_total", 1.0, labels)
        self.metric_collector.record_histogram("ai_operation_duration_seconds", duration, labels)
        
        if not success:
            self.metric_collector.record_counter("ai_operation_errors_total", 1.0, labels)
        
        if accuracy is not None:
            self.metric_collector.record_histogram("ai_operation_accuracy", accuracy, labels)
    
    def record_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Record system resource metrics."""
        self.metric_collector.record_gauge("system_cpu_usage_percent", cpu_usage)
        self.metric_collector.record_gauge("system_memory_usage_percent", memory_usage)
        self.metric_collector.record_gauge("system_disk_usage_percent", disk_usage)
    
    def setup_default_alerts(self):
        """Setup default alert rules for common issues."""
        # High error rate alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="http_errors_total",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.HIGH,
            duration=60
        ))
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_usage_percent",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.MEDIUM,
            duration=120
        ))
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_usage_percent",
            condition="gt",
            threshold=85.0,
            severity=AlertSeverity.HIGH,
            duration=60
        ))
    
    def check_alerts_if_needed(self):
        """Check alerts if enough time has passed since last check."""
        current_time = time.time()
        if current_time - self.last_alert_check >= self.alert_check_interval:
            self.alert_manager.check_alerts()
            self.last_alert_check = current_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        summary = {
            "total_metrics": len(self.metric_collector.metrics),
            "active_alerts": len(self.alert_manager.active_alerts),
            "metrics": {}
        }
        
        for key, metric in self.metric_collector.metrics.items():
            if metric["type"] == MetricType.HISTOGRAM:
                stats = self.metric_collector.get_histogram_stats(key.split("{")[0])
                summary["metrics"][key] = stats
            else:
                summary["metrics"][key] = {
                    "type": metric["type"].value,
                    "value": metric.get("value", 0.0)
                }
        
        return summary


# Global monitoring system instance
monitoring_system = MonitoringSystem()
