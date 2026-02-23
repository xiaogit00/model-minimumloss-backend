#!/bin/bash
# deploy-backend.sh

DROPLET_IP="206.189.156.163"
DEPLOY_USER="deploy"
BACKEND_DIR="/home/deploy/model-minimumloss-backend"

echo "🚀 Deploying backend..."

ssh $DEPLOY_USER@$DROPLET_IP "
  cd $BACKEND_DIR

  echo '📥 Pulling latest code...'
  git pull origin main

  echo '🔄 Restarting FastAPI...'
  sudo systemctl restart fastapi
"

echo "🎉 Backend deployment complete!"