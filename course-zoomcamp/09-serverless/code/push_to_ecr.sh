# !bin/bash

REGION=us-east-2
REGISTRY_NAME=clothing-model-tflite-images
REGISTRY_URL=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY_NAME}
TAG=clothing-model-xception-v4-001
REMOTE_URI=${REGISTRY_URL}:${TAG}