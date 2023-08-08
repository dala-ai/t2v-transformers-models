#!/bin/bash

help_msg="Usage: ./build.sh <model>"

# Quick validations
if [[ "-h" == "$1" ]]; then
  echo $help_msg
  exit 0
fi

if [[ -z "$1" ]]; then
  echo "You did not set a model name"
  exit 0
fi

MODEL_NAME=$1

DATE=$(date -u +"%Y-%m-%dT%H.%M.%SZ")
TAG=$(echo -n "${MODEL_NAME}.${DATE}" | sed -e "s/\//_/g")
IMAGE_REPO="560642106477.dkr.ecr.eu-west-2.amazonaws.com"

docker build -t dala-model:$TAG \
 -t $IMAGE_REPO/dala-models:$TAG \
 --build-arg MODEL_NAME=$MODEL_NAME \
 .