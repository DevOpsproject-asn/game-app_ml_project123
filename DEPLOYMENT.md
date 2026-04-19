"""Deployment guide for Game ML Platform"""

# Deployment Guide

## Local Development

### 1. Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Train Models
```bash
python pipelines/training_pipeline.py
```

### 3. Start API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Test
```bash
pytest tests/ -v
```

## Docker Deployment

### 1. Using Docker Compose (Recommended)
```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### 2. Manual Docker Setup
```bash
# Build image
docker build -t game-ml-platform:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  game-ml-platform:latest
```

## Production Deployment

### Kubernetes Deployment

1. **Build and push Docker image**
```bash
docker build -t your-registry/game-ml-platform:v1.0.0 .
docker push your-registry/game-ml-platform:v1.0.0
```

2. **Create Kubernetes manifests** (example: deployment.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: game-ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: game-ml-api
  template:
    metadata:
      labels:
        app: game-ml-api
    spec:
      containers:
      - name: api
        image: your-registry/game-ml-platform:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

3. **Deploy to Kubernetes**
```bash
kubectl apply -f deployment.yaml
```

## Cloud Deployment

### AWS (example)
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker tag game-ml-platform:latest your-account.dkr.ecr.us-east-1.amazonaws.com/game-ml-platform:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/game-ml-platform:latest

# Deploy to ECS/Fargate
# (Use AWS Console or Terraform)
```

### Google Cloud Platform
```bash
# Push to GCR
docker tag game-ml-platform:latest gcr.io/your-project/game-ml-platform:latest
docker push gcr.io/your-project/game-ml-platform:latest

# Deploy to Cloud Run
gcloud run deploy game-ml-platform \
  --image gcr.io/your-project/game-ml-platform:latest \
  --platform managed \
  --region us-central1 \
  --port 8000
```

## Monitoring & Logging

### MLflow Monitoring
- Access at http://localhost:5000 or your MLflow server URL
- View experiments, runs, and model metrics

### Application Logs
- Local: Check `logs/game_ml.log`
- Docker: `docker-compose logs api`
- Production: Use CloudWatch, Stackdriver, or similar

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check models
curl http://localhost:8000/api/v1/health/models
```

## Scaling Considerations

1. **API Server**
   - Use multiple replicas behind load balancer
   - Configure auto-scaling based on CPU/memory
   - Set appropriate resource limits

2. **Database**
   - Use read replicas for scaling reads
   - Configure connection pooling
   - Regular backups

3. **Model Serving**
   - Cache model in memory
   - Use model batching for efficiency
   - Consider dedicated model serving (TensorFlow Serving, Seldon)

4. **Monitoring**
   - Track API response times
   - Monitor model prediction latency
   - Alert on performance degradation

## Troubleshooting

### Issue: API won't start
```bash
# Check logs
docker-compose logs api

# Verify ports
lsof -i :8000

# Check environment variables
docker-compose config
```

### Issue: Database connection fails
```bash
# Test connection
psql -U game_user -h localhost -d game_analytics

# Check PostgreSQL status
docker-compose ps postgres
```

### Issue: Models not loading
```bash
# Check model files exist
ls -la models/

# Verify model paths in code
grep -r "model_path" src/
```

## Rollback Procedure

```bash
# If using Docker
docker run -p 8000:8000 your-registry/game-ml-platform:v1.0.0

# If using Kubernetes
kubectl rollout undo deployment/game-ml-api
```

## Performance Tuning

1. **API Server**
   - Increase workers: `--workers 4`
   - Tune thread pool size
   - Enable response compression

2. **Model Loading**
   - Load models on startup (not per request)
   - Use model caching
   - Consider quantization for faster inference

3. **Database**
   - Add indexes on frequently queried columns
   - Use connection pooling
   - Enable query caching

See README.md for more information.
