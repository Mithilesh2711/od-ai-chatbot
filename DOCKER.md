# Docker Setup Documentation

This document provides comprehensive instructions for running the AI Chatbot project in Docker containers on Ubuntu VMs.

## 📋 Prerequisites

- Docker Engine 20.10+ 
- Docker Compose v2.0+
- Ubuntu 20.04+ (or compatible Linux distribution)
- Minimum 4GB RAM, 2 CPU cores recommended

## 🏗️ Project Structure

```
od-ai-chatbot/
├── Dockerfile              # Main application container
├── docker-compose.yml      # Multi-container orchestration
├── .dockerignore          # Files to exclude from Docker context
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── main.py               # FastAPI application entry point
└── ... (other project files)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd od-ai-chatbot

# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

### 2. Environment Configuration

Edit the `.env` file with your actual credentials:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_actual_openai_api_key

# Required: MongoDB Connection
MONGODB_URL=mongodb://username:password@hostname:port/database

# Required: Qdrant Vector Database
QDRANT_URL=https://your-qdrant-instance.com:6333
QDRANT_API_KEY=your_qdrant_api_key

# Security: Change this in production
JWT_SECRET_KEY=your_secure_secret_key_here
```

### 3. Build and Run

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔧 Docker Configuration Details

### Application Container

The main application runs in a Python 3.11 container with:

- **Base Image**: `python:3.11-slim` (Ubuntu-based)
- **Port**: 8000 (exposed to host)
- **Health Check**: Built-in FastAPI health endpoints
- **Development**: Volume mounting for hot-reload
- **Production**: Optimized for performance

### Redis Container

- **Image**: `redis:7-alpine`
- **Port**: 6379 (exposed for caching)
- **Purpose**: Session management and caching

### Volume Mounting

**Development Mode**:
- Current directory mounted to `/app` for live code changes
- Python cache directories excluded via `.dockerignore`

**Production Mode**:
- Only necessary directories mounted
- Optimized for performance

## 🔐 Security Configuration

### Environment Variables

All sensitive configuration is handled via environment variables:

- **Never commit** `.env` files to version control
- Use `.env.example` as template for new deployments
- Rotate JWT secret keys regularly

### Network Security

- Internal container communication on isolated network
- Only essential ports exposed to host
- Redis accessible only via internal network

## 📊 Monitoring and Health Checks

### Application Health

```bash
# Check API health
curl http://localhost:8000/health

# Check application status
curl http://localhost:8000/
```

### Container Health

```bash
# View container status
docker-compose ps

# View resource usage
docker stats

# View application logs
docker-compose logs od-ai-chatbot

# View Redis logs
docker-compose logs redis
```

## 🛠️ Development Workflow

### Hot Reload Development

```bash
# Start in development mode with volume mounting
docker-compose up

# Code changes are reflected immediately
# No container restart needed for code changes
```

### Debugging

```bash
# Access container shell
docker-compose exec od-ai-chatbot bash

# Run Python REPL in container
docker-compose exec od-ai-chatbot python

# Test individual components
docker-compose exec od-ai-chatbot python -c "import main; print('Import successful')"
```

## 🚀 Production Deployment

### Production Compose File

For production, create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  od-ai-chatbot:
    build: .
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - redis
    networks:
      - backend
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    networks:
      - backend

networks:
  backend:
    driver: bridge
```

### Production Deployment Commands

```bash
# Build and deploy to production
docker-compose -f docker-compose.prod.yml up -d --build

# Scale application (if load balancing needed)
docker-compose -f docker-compose.prod.yml up -d --scale od-ai-chatbot=3
```

### Systemd Service (Optional)

Create `/etc/systemd/system/od-ai-chatbot.service`:

```ini
[Unit]
Description=OD AI Chatbot Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/od-ai-chatbot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable od-ai-chatbot
sudo systemctl start od-ai-chatbot
```

## 🧪 Testing

### Unit Tests in Container

```bash
# Run tests inside container
docker-compose exec od-ai-chatbot python -m pytest tests/

# Run specific test
docker-compose exec od-ai-chatbot python -m pytest tests/test_main.py
```

### Integration Testing

```bash
# Test API endpoints
curl -X GET http://localhost:8000/health

# Test vector service
curl -X POST http://localhost:8000/vector/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Check port usage
sudo netstat -tulpn | grep :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

**Memory Issues**:
```bash
# Check container memory usage
docker stats

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

**Permission Issues**:
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Ensure Docker socket permissions
sudo chmod 666 /var/run/docker.sock
```

### Log Analysis

```bash
# View real-time logs
docker-compose logs -f

# View logs with timestamps
docker-compose logs -f -t

# Search logs for errors
docker-compose logs | grep -i error
```

## 📈 Performance Optimization

### Build Optimization

- Use multi-stage builds for smaller images
- Leverage Docker layer caching
- Minimize image layers

### Runtime Optimization

- Enable connection pooling
- Configure Redis caching
- Monitor memory usage

### Resource Limits

```yaml
# Add to docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '1.0'
```

## 🔄 Updates and Maintenance

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Database Migrations

```bash
# Run migrations (if applicable)
docker-compose exec od-ai-chatbot python manage.py migrate

# Or use custom migration scripts
docker-compose exec od-ai-chatbot python scripts/migrate.py
```

### Backup and Restore

```bash
# Backup application data
docker-compose exec od-ai-chatbot python scripts/backup.py

# Restore from backup
docker-compose exec od-ai-chatbot python scripts/restore.py backup_file.tar.gz
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project Repository](https://github.com/Mithilesh2711/od-ai-chatbot.git)

---

For support or questions, please refer to the project repository or create an issue.
