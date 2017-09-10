#!/usr/bin/env bash
cd ..
#source .env/bin/activate
REL_HOME=$(pwd)
export PYTHONPATH="$REL_HOME":$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
cd ReferringRelationships
