#!/usr/bin/env bash
export REL_HOME="$( cd .. && pwd )"
export PYTHONPATH="$PYTHONPATH:$REL_HOME"
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
echo "Environment variables set!"
