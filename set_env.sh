#!/usr/bin/env bash
cd ..
REL_HOME=$(pwd)
export $PYTHONPATH="$REL_HOME":$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/