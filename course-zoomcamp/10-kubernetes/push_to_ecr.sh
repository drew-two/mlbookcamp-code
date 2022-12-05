# !bin/bash

ACCOUNT_ID=183665471551
REGION=us-east-2
REGISTRY_NAME=mlzoomcamp-images
REGISTRY_URL=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY_NAME}
TAG=clothing-model-xception-v4-001
REMOTE_URI=${REGISTRY_URL}:${TAG}

GATEWAY_LOCAL=zoomcamp-10-gateway:002
GATEWAY_REMOTE=${REGISTRY_URL}:zoomcamp-10-gateway-002
docker tag ${GATEWAY_LOCAL} ${GATEWAY_REMOTE}

MODEL_LOCAL=zoomcamp-10-model:xception-v4-001
MODEL_REMOTE=${REGISTRY_URL}:zoomcamp-10-model-xception-v4-001
docker tag ${MODEL_LOCAL} ${MODEL_REMOTE}

docker push ${GATEWAY_REMOTE}
docker push ${MODEL_REMOTE}