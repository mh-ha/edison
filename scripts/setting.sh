#!/bin/bash

# Change directory to edison_storage/edison
cd ~/edison_storage/edison

# Create a conda environment from the conda.yaml file
conda env create -f conda.yaml
conda init bash
source ~/.bashrc
conda activate edison

# Download the spaCy model en_core_web_sm
python -m spacy download en_core_web_sm