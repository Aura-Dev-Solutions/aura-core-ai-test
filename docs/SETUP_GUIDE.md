# Guía de Setup y Deployment

##  Quick Start

### Prerrequisitos
- Python 3.11+
- Docker y Docker Compose
- Git

### Setup Rápido
```bash
# Clonar repositorio
git clone https://github.com/Aura-Dev-Solutions/aura-core-ai-test.git
cd aura-core-ai-test

# Setup completo de desarrollo
make setup-dev

# Ejecutar aplicación
make run-dev
```

## 🛠️ Setup Detallado

### 1. Entorno de Desarrollo

#### Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

#### Instalar dependencias
```bash
# Dependencias de desarrollo
pip install -e .[dev]

# Modelos de spaCy
python -m spacy download en_core_web_sm

# Pre-commit hooks
pre-commit install
```

#### Configurar variables de entorno
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar configuración
nano .env
```

### 2. Base de Datos

#### PostgreSQL Local
```bash
# Instalar PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
brew install postgresql                             # macOS

# Crear base de datos
sudo -u postgres createdb aura_docs
sudo -u postgres createuser aura_user
sudo -u postgres psql -c "ALTER USER aura_user WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aura_docs TO aura_user;"
```

#### Redis Local
```bash
# Instalar Redis
sudo apt-get install redis-server  # Ubuntu
brew install redis                 # macOS

# Iniciar Redis
redis-server
```

### 3. Docker Setup

#### Desarrollo con Docker
```bash
# Construir imagen de desarrollo
make docker-build-dev

# Ejecutar con Docker Compose
make docker-run

# Ver logs
make docker-logs

# Acceder al container
make docker-shell
```

#### Producción con Docker
```bash
# Construir imagen de producción
make docker-build

# Ejecutar en producción
docker-compose -f docker-compose.prod.yml up -d
```

##  Testing

### Ejecutar Tests
```bash
# Todos los tests
make test

# Solo tests unitarios
make test-unit

# Tests con coverage
pytest --cov=src --cov-report=html

# Tests rápidos (sin slow tests)
make test-fast
```

### Tests de Integración
```bash
# Tests de integración
make test-integration

# Tests end-to-end
make test-e2e
```

##  Comandos de Desarrollo

### Calidad de Código
```bash
# Formatear código
make format

# Linting
make lint

# Type checking
make type-check

# Todo junto
make ci-quality
```

### Base de Datos
```bash
# Crear migración
make db-migration MESSAGE="Add new table"

# Aplicar migraciones
make db-upgrade

# Rollback
make db-downgrade

# Reset completo
make db-reset
```

### Utilidades
```bash
# Shell interactivo
make shell

# Descargar modelos de IA
make download-models

# Benchmark de performance
make benchmark

# Security check
make security-check
```

##  Docker Compose Profiles

### Desarrollo Básico
```bash
docker-compose up -d
# Incluye: app, postgres, redis
```

### Con Monitoreo
```bash
docker-compose --profile monitoring up -d
# Incluye: app, postgres, redis, prometheus, grafana
```

### Producción
```bash
docker-compose --profile production up -d
# Incluye: app, postgres, redis, nginx
```

##  Configuración de Producción

### Variables de Entorno Críticas
```bash
# Seguridad
SECRET_KEY=your-super-secret-key-here
ENVIRONMENT=production
DEBUG=false

# Base de Datos
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
REDIS_URL=redis://redis-host:6379/0

# Límites
MAX_FILE_SIZE=104857600  # 100MB
MAX_WORKERS=8

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Nginx Configuration
```nginx
upstream aura_app {
    server app:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://aura_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        proxy_pass http://aura_app/health;
        access_log off;
    }
}
```

### SSL/TLS con Let's Encrypt
```bash
# Instalar certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d your-domain.com

# Auto-renovación
sudo crontab -e
# Agregar: 0 12 * * * /usr/bin/certbot renew --quiet
```

##  Monitoreo

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aura-app'
    static_configs:
      - targets: ['app:9090']
```

### Grafana Dashboards
- **Application Metrics**: Request rate, latency, errors
- **Document Processing**: Processing time, success rate, queue size
- **System Metrics**: CPU, memory, disk usage
- **Database Metrics**: Connection pool, query performance

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db

# Redis health
curl http://localhost:8000/health/redis
```

## 🔒 Seguridad

### Configuración de Firewall
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Backup Strategy
```bash
# Database backup
pg_dump -h localhost -U aura_user aura_docs > backup.sql

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U aura_user aura_docs | gzip > /backups/aura_docs_$DATE.sql.gz
find /backups -name "aura_docs_*.sql.gz" -mtime +7 -delete
```

### Log Rotation
```bash
# /etc/logrotate.d/aura-app
/var/log/aura-app/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 aura-app aura-app
    postrotate
        systemctl reload aura-app
    endscript
}
```

##  Deployment Strategies

### Blue-Green Deployment
```bash
# Deploy nueva versión
docker-compose -f docker-compose.blue.yml up -d

# Verificar health
curl http://blue.internal/health

# Switch traffic
# (Update load balancer configuration)

# Cleanup old version
docker-compose -f docker-compose.green.yml down
```

### Rolling Updates
```bash
# Kubernetes deployment
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/aura-app
kubectl rollout history deployment/aura-app
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: make ci-test
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          docker build -t aura-app:${{ github.sha }} .
          docker push aura-app:${{ github.sha }}
          # Update production deployment
```

##  Troubleshooting

### Problemas Comunes

#### Error de Conexión a Base de Datos
```bash
# Verificar conexión
pg_isready -h localhost -p 5432

# Verificar logs
docker-compose logs postgres

# Recrear base de datos
make db-reset
```

#### Error de Memoria
```bash
# Verificar uso de memoria
docker stats

# Ajustar límites
# En docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### Problemas de Performance
```bash
# Profiling
python -m cProfile -o profile.stats your_script.py

# Monitoring
htop
iotop
```

### Logs Útiles
```bash
# Application logs
docker-compose logs -f app

# Database logs
docker-compose logs postgres

# System logs
journalctl -u aura-app -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

##  Soporte

### Contactos
- **Email técnico**: otorres@auraresearch.ai, igutierrez@auraresearch.ai
- **Issues**: GitHub Issues en el repositorio
- **Documentación**: `/docs` en el repositorio

### Recursos Adicionales
- **API Documentation**: http://localhost:8000/docs
- **Monitoring Dashboard**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)
