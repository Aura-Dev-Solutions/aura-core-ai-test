#!/usr/bin/env python3
"""
🚀 Script de Inicialización - Aura Document Analyzer
Sistema de Análisis de Documentos Escalable con IA

Este script inicializa completamente el sistema y verifica que todos
los componentes estén funcionando correctamente.
"""

import requests
import time
import json
import subprocess
import sys
import os
from pathlib import Path

def print_header(title: str):
    """Imprimir encabezado de sección"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_status(message: str, success: bool = True):
    """Imprimir estado con formato"""
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")

def check_server_status():
    """Verificar si el servidor está corriendo"""
    try:
        response = requests.get("http://localhost:8000/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_server(max_attempts=30):
    """Esperar a que el servidor esté listo"""
    print("⏳ Esperando a que el servidor se inicialice...")
    
    for attempt in range(max_attempts):
        if check_server_status():
            print_status("Servidor iniciado correctamente")
            return True
        
        print(f"   Intento {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    return False

def verify_system_components():
    """Verificar todos los componentes del sistema"""
    print_header("VERIFICACIÓN DE COMPONENTES")
    
    components = [
        ("Health Check", "http://localhost:8000/api/v1/health"),
        ("Estadísticas", "http://localhost:8000/api/v1/stats"),
        ("Documentación", "http://localhost:8000/docs"),
    ]
    
    all_ok = True
    
    for name, url in components:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print_status(f"{name}: OK")
            else:
                print_status(f"{name}: Error {response.status_code}", False)
                all_ok = False
        except Exception as e:
            print_status(f"{name}: Error de conexión", False)
            all_ok = False
    
    return all_ok

def test_ai_functionality():
    """Probar funcionalidad de IA"""
    print_header("PRUEBA DE FUNCIONALIDAD IA")
    
    test_text = """
    CONTRATO DE DESARROLLO DE SOFTWARE
    Entre Aura Research S.A. de C.V. y DevCorp Technologies Inc.
    Monto: $50,000 USD
    Contacto: ana.martinez@devcorp.com
    Fecha: 2025-01-18
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/ai/analyze",
            json={
                "text": test_text.strip(),
                "include_classification": True,
                "include_ner": True,
                "include_embeddings": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("Análisis de IA: OK")
            print(f"   📊 Clasificación: {data.get('classification', {}).get('category', 'N/A')}")
            print(f"   🏷️  Entidades: {len(data.get('entities', []))} encontradas")
            print(f"   ⏱️  Tiempo: {data.get('processing_time', 0):.3f}s")
            return True
        else:
            print_status("Análisis de IA: Error", False)
            return False
            
    except Exception as e:
        print_status(f"Análisis de IA: Error - {e}", False)
        return False

def show_system_info():
    """Mostrar información del sistema"""
    print_header("INFORMACIÓN DEL SISTEMA")
    
    try:
        # Health check
        health_response = requests.get("http://localhost:8000/api/v1/health")
        health_data = health_response.json()
        
        print("🏥 Estado de Salud:")
        print(f"   • Estado general: {health_data.get('status', 'unknown')}")
        print(f"   • Versión: {health_data.get('version', 'unknown')}")
        
        components = health_data.get('components', {})
        for component, status in components.items():
            icon = "✅" if status == "healthy" else "❌"
            print(f"   • {component}: {icon} {status}")
        
        # Estadísticas
        stats_response = requests.get("http://localhost:8000/api/v1/stats")
        stats_data = stats_response.json()
        
        print("\n📊 Estadísticas:")
        print(f"   • Documentos procesados: {stats_data.get('processed_documents', 0)}")
        print(f"   • Tiempo promedio: {stats_data.get('average_processing_time', 0):.3f}s")
        print(f"   • Throughput: {stats_data.get('documents_per_minute', 0):.0f} docs/min")
        
        ai_models = stats_data.get('ai_models_loaded', {})
        print("\n🤖 Modelos de IA:")
        for model, loaded in ai_models.items():
            icon = "✅" if loaded else "❌"
            print(f"   • {model}: {icon}")
            
    except Exception as e:
        print_status(f"Error obteniendo información: {e}", False)

def show_access_info():
    """Mostrar información de acceso"""
    print_header("INFORMACIÓN DE ACCESO")
    
    print("🌐 URLs del Sistema:")
    print("   • API Principal: http://localhost:8000/")
    print("   • Documentación: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • Health Check: http://localhost:8000/api/v1/health")
    print("   • Estadísticas: http://localhost:8000/api/v1/stats")
    
    print("\n📋 Endpoints Principales:")
    print("   • POST /api/v1/documents/upload - Subir documentos")
    print("   • POST /api/v1/ai/analyze - Análisis con IA")
    print("   • POST /api/v1/search/semantic - Búsqueda semántica")
    print("   • GET  /api/v1/documents - Listar documentos")
    
    print("\n🔧 Comandos Útiles:")
    print("   • Detener servidor: pkill -f uvicorn")
    print("   • Ver logs: tail -f aura_server.log")
    print("   • Reiniciar: python inicializar_sistema.py")

def main():
    """Función principal de inicialización"""
    print("🚀 INICIALIZANDO AURA DOCUMENT ANALYZER")
    print("Sistema de Análisis de Documentos Escalable con IA")
    print("="*80)
    
    # Verificar si ya está corriendo
    if check_server_status():
        print_status("El servidor ya está corriendo")
    else:
        print("🔄 Iniciando servidor...")
        
        # Cambiar al directorio del proyecto
        os.chdir(Path(__file__).parent)
        
        # Iniciar servidor en background
        try:
            subprocess.Popen([
                "bash", "-c", 
                "source venv/bin/activate && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info > aura_server.log 2>&1"
            ])
            
            # Esperar a que se inicialice
            if not wait_for_server():
                print_status("Error: No se pudo iniciar el servidor", False)
                return False
                
        except Exception as e:
            print_status(f"Error iniciando servidor: {e}", False)
            return False
    
    # Verificar componentes
    if not verify_system_components():
        print_status("Algunos componentes tienen problemas", False)
        return False
    
    # Probar IA
    if not test_ai_functionality():
        print_status("Funcionalidad de IA tiene problemas", False)
        return False
    
    # Mostrar información del sistema
    show_system_info()
    
    # Mostrar información de acceso
    show_access_info()
    
    # Resultado final
    print_header("INICIALIZACIÓN COMPLETADA")
    print("🎉 ¡Sistema Aura Document Analyzer iniciado exitosamente!")
    print("✅ Todos los componentes están funcionando correctamente")
    print("🚀 El sistema está listo para procesar documentos")
    print("\n💡 Próximos pasos:")
    print("   1. Visita http://localhost:8000/docs para ver la documentación")
    print("   2. Prueba subir un documento usando la API")
    print("   3. Explora las funcionalidades de IA disponibles")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Inicialización cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
