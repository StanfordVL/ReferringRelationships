#!/bin/bash

mkdir -p data
cd data

echo "Downloding VRD..."
wget https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/vrd.zip
unzip vrd.zip
rm vrd.zip

echo "Downloading CLEVR..."
wget https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/clevr.zip
unzip clevr
rm clevr.zip

echo "Downloading Visual Genome..."
wget https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/visualgenome.zip
unzip visualgenome.zip
rm visualgenome.zip

cd ..
