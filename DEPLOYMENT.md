# Production Deployment Guide

## Overview

This guide covers deploying the RAG System to a production environment with proper security, monitoring, and scalability.

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Docker**: 24.0.0+
- **Docker Compose**: 2.20.0+

### Optional

- NVIDIA GPU for Ollama (significantly improves performance)
- Domain name with SSL certificate
- Cloud storage for backups

## Pre-Deployment Checklist

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (if using GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Clone Repository

```bash
cd /opt
sudo git clone <repository-url> rag-system
cd rag-system
sudo chown -R $USER:$USER /opt/rag-system
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with production values
nano .env
```

**Critical Environment Variables:**

```bash
# Database - Use strong passwords
POSTGRES_USER=rag_prod_user
POSTGRES_PASSWORD=<generate-strong-password>
POSTGRES_DB=rag_production

# Google AI
GOOGLE_API_KEY=<your-production-api-key>

# Security
ALLOWED_ORIGINS=https://yourdomain.com
JWT_SECRET_KEY=<generate-random-secret>
API_KEY_SALT=<generate-random-salt>

# Resource Limits
MAX_FILE_SIZE=52428800  # 50MB
MAX_WORKERS=4
```

### 4. Generate Secrets

```bash
# Generate secure passwords
openssl rand -base64 32  # For POSTGRES_PASSWORD
openssl rand -hex 32     # For JWT_SECRET_KEY
openssl rand -hex 16     # For API_KEY_SALT
```

## Deployment Steps

### 1. Production Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    restart: always
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  ui:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  postgres:
    restart: always
    volumes:
      - /data/rag/postgres:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  milvus:
    restart: always
    volumes:
      - /data/rag/milvus:/var/lib/milvus
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 2. Build & Deploy

```bash
# Create data directories
sudo mkdir -p /data/rag/{postgres,milvus,minio,etcd}
sudo chown -R $USER:$USER /data/rag

# Build images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Test API
curl http://localhost:8000/docs

# Test UI
curl http://localhost:8501
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS (if using reverse proxy)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Deny direct access to services
sudo ufw deny 8000/tcp
sudo ufw deny 8501/tcp
sudo ufw deny 5432/tcp
sudo ufw deny 19530/tcp

# Enable firewall
sudo ufw enable
```

### 2. Reverse Proxy (Nginx)

```bash
# Install Nginx
sudo apt install nginx

# Create configuration
sudo nano /etc/nginx/sites-available/rag-system
```

**Nginx Configuration:**

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS Configuration
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # UI (Streamlit)
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long-running requests
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    # Health check endpoint (no auth required)
    location /api/health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 3. SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal (already configured by certbot)
sudo certbot renew --dry-run
```

### 4. Database Security

```bash
# Access PostgreSQL
docker exec -it rag_postgres psql -U rag_prod_user -d rag_production

# Create read-only user for monitoring
CREATE USER monitor WITH PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE rag_production TO monitor;
GRANT USAGE ON SCHEMA public TO monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitor;

# Enable SSL (optional)
# Mount SSL certificates in docker-compose.prod.yml
```

## Monitoring Setup

### 1. System Monitoring

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Check system resources
htop

# Monitor disk I/O
sudo iotop

# Monitor network
sudo nethogs
```

### 2. Application Monitoring

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: rag_prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - rag_network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: rag_grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - rag_network
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  rag_network:
    external: true
```

**Prometheus Configuration** (`monitoring/prometheus.yml`):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 3. Log Management

```bash
# Install Loki (optional)
# Or use external service like Papertrail, Loggly

# Configure log rotation
sudo nano /etc/docker/daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

```bash
sudo systemctl restart docker
```

## Backup Strategy

### 1. Automated Database Backups

Create `/opt/rag-system/backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/data/rag/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec rag_postgres pg_dump -U rag_prod_user rag_production | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Backup Milvus metadata
tar -czf $BACKUP_DIR/milvus_$DATE.tar.gz /data/rag/milvus

# Remove old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Upload to S3 (optional)
# aws s3 sync $BACKUP_DIR s3://your-bucket/rag-backups/

echo "Backup completed: $DATE"
```

```bash
# Make executable
chmod +x /opt/rag-system/backup.sh

# Add to crontab (daily at 2 AM)
sudo crontab -e
0 2 * * * /opt/rag-system/backup.sh >> /var/log/rag-backup.log 2>&1
```

### 2. Volume Snapshots

```bash
# Create snapshot script
#!/bin/bash
sudo rsync -av /data/rag/ /backup/rag-$(date +%Y%m%d)/
```

## Scaling

### 1. Horizontal Scaling (API)

```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      replicas: 3
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.yourdomain.com`)"
```

### 2. Vertical Scaling

```bash
# Increase resources in docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
```

### 3. Database Scaling

```bash
# Enable connection pooling
services:
  api:
    environment:
      - DATABASE_POOL_SIZE=20
      - DATABASE_MAX_OVERFLOW=10
```

## Maintenance

### Regular Tasks

```bash
# Weekly
- Review logs for errors
- Check disk space
- Verify backups

# Monthly
- Update Docker images
- Review security updates
- Optimize database

# Quarterly
- Performance audit
- Security audit
- Disaster recovery test
```

### Update Procedure

```bash
# 1. Backup current state
./backup.sh

# 2. Pull latest code
git pull

# 3. Review changes
git log -5

# 4. Build new images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# 5. Rolling update
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 6. Verify health
curl http://localhost:8000/health

# 7. Monitor logs
docker-compose logs -f api
```

## Disaster Recovery

### Recovery Procedure

```bash
# 1. Stop all services
docker-compose down

# 2. Restore database
gunzip -c /data/rag/backups/postgres_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i rag_postgres psql -U rag_prod_user rag_production

# 3. Restore volumes
sudo tar -xzf /data/rag/backups/milvus_YYYYMMDD_HHMMSS.tar.gz -C /

# 4. Restart services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 5. Verify
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase limits in docker-compose.prod.yml
   ```

2. **Slow Performance**
   ```bash
   # Check database connections
   docker exec -it rag_postgres psql -U rag_prod_user -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check Milvus performance
   curl http://localhost:9091/metrics
   ```

3. **Database Connection Issues**
   ```bash
   # Restart PostgreSQL
   docker-compose restart postgres
   
   # Check logs
   docker-compose logs postgres
   ```

## Support

For production support:
- Email: ops@example.com
- Slack: #rag-system-ops
- On-call: Use PagerDuty escalation

## Compliance

### Data Privacy
- Ensure GDPR compliance
- Implement data retention policies
- Enable audit logging

### Security Standards
- Follow OWASP guidelines
- Regular penetration testing
- Security patch management

---

**Last Updated**: 2024
**Version**: 1.0.0