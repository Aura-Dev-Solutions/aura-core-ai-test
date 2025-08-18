"""
Monitor de performance para el sistema de análisis de documentos.

Este módulo proporciona herramientas para monitorear y medir el rendimiento
de diferentes operaciones del sistema.
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from datetime import datetime, timedelta


class PerformanceMonitor:
    """
    Monitor de performance para medir tiempos de ejecución y métricas.
    
    Permite medir operaciones individuales y generar estadísticas agregadas.
    """
    
    def __init__(self):
        """Inicializar el monitor de performance."""
        self._measurements: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """
        Context manager para medir tiempo de una operación.
        
        Args:
            operation_name: Nombre de la operación a medir
            
        Usage:
            with monitor.measure("document_processing"):
                # código a medir
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_measurement(operation_name, duration)
    
    def start_measurement(self, operation_name: str):
        """
        Iniciar medición de una operación.
        
        Args:
            operation_name: Nombre de la operación
        """
        self._start_times[operation_name] = time.time()
    
    def end_measurement(self, operation_name: str) -> Optional[float]:
        """
        Finalizar medición de una operación.
        
        Args:
            operation_name: Nombre de la operación
            
        Returns:
            Duración de la operación en segundos, o None si no se inició
        """
        if operation_name not in self._start_times:
            return None
        
        duration = time.time() - self._start_times[operation_name]
        del self._start_times[operation_name]
        
        self.record_measurement(operation_name, duration)
        return duration
    
    def record_measurement(self, operation_name: str, duration: float):
        """
        Registrar una medición de tiempo.
        
        Args:
            operation_name: Nombre de la operación
            duration: Duración en segundos
        """
        if operation_name not in self._measurements:
            self._measurements[operation_name] = []
        
        self._measurements[operation_name].append(duration)
    
    def increment_counter(self, counter_name: str, increment: int = 1):
        """
        Incrementar un contador.
        
        Args:
            counter_name: Nombre del contador
            increment: Cantidad a incrementar (default: 1)
        """
        if counter_name not in self._counters:
            self._counters[counter_name] = 0
        
        self._counters[counter_name] += increment
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de performance.
        
        Args:
            operation_name: Nombre de operación específica, o None para todas
            
        Returns:
            Diccionario con estadísticas
        """
        if operation_name:
            return self._get_operation_stats(operation_name)
        
        # Estadísticas de todas las operaciones
        stats = {
            'operations': {},
            'counters': dict(self._counters),
            'summary': {
                'total_operations': len(self._measurements),
                'total_measurements': sum(len(measurements) for measurements in self._measurements.values())
            }
        }
        
        for op_name in self._measurements:
            stats['operations'][op_name] = self._get_operation_stats(op_name)
        
        return stats
    
    def _get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Obtener estadísticas para una operación específica.
        
        Args:
            operation_name: Nombre de la operación
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        if operation_name not in self._measurements:
            return {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0,
                'median_time': 0
            }
        
        measurements = self._measurements[operation_name]
        
        return {
            'count': len(measurements),
            'total_time': sum(measurements),
            'avg_time': statistics.mean(measurements),
            'min_time': min(measurements),
            'max_time': max(measurements),
            'median_time': statistics.median(measurements),
            'std_dev': statistics.stdev(measurements) if len(measurements) > 1 else 0
        }
    
    def reset(self):
        """Resetear todas las mediciones y contadores."""
        self._measurements.clear()
        self._counters.clear()
        self._start_times.clear()
    
    def reset_operation(self, operation_name: str):
        """
        Resetear mediciones de una operación específica.
        
        Args:
            operation_name: Nombre de la operación a resetear
        """
        if operation_name in self._measurements:
            del self._measurements[operation_name]
        
        if operation_name in self._start_times:
            del self._start_times[operation_name]
    
    def get_throughput(self, operation_name: str, time_window: Optional[timedelta] = None) -> float:
        """
        Calcular throughput (operaciones por segundo) para una operación.
        
        Args:
            operation_name: Nombre de la operación
            time_window: Ventana de tiempo a considerar (None para todas)
            
        Returns:
            Throughput en operaciones por segundo
        """
        if operation_name not in self._measurements:
            return 0.0
        
        measurements = self._measurements[operation_name]
        if not measurements:
            return 0.0
        
        # Si no hay ventana de tiempo, usar todas las mediciones
        if time_window is None:
            total_time = sum(measurements)
            return len(measurements) / total_time if total_time > 0 else 0.0
        
        # Para ventana de tiempo específica, necesitaríamos timestamps
        # Por simplicidad, usar todas las mediciones
        total_time = sum(measurements)
        return len(measurements) / total_time if total_time > 0 else 0.0
