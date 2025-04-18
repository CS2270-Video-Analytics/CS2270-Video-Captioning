#!/bin/bash

echo 'export OLLAMA_MODELS=/oscar/data/shared/ollama_models' >> ~/.bashrc

source ~/.bashrc

# Default model name
model_name="llama3.2-vision:11b"

# Parse command-line arguments
while getopts "m:" opt; do
  case ${opt} in
    m )
      model_name=$OPTARG  # Overwrite model_name with the value passed by -m
      ;;
    \? )
      echo "Usage: $0 [-m model_name]"
      exit 1
      ;;
  esac
done

module load ollama
ollama serve