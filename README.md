# A-High-Resolution-500-m-Dataset-for-Mapping-Key-Agronomic-Growth-Stages-of-Maize-in-Northeast-China
# Maize Phenology Mapping for Northeast China

[![DOI](10.5281/zenodo.17666835) 
This repository contains the source code for generating the **High-Resolution (500-m) Maize Phenology Dataset (2001-2024)** for Northeast China, as described in the accompanying manuscript (under review).

The dataset captures eight key agronomic growth stages of maize: emergence, three-leaf, seven-leaf, jointing, tasseling, silking, milk stage, and maturity.

## 🗂️ Project Description

This project aims to reconstruct a continuous, high-resolution maize phenology dataset by integrating multi-source satellite data and meteorological data using machine learning techniques (XGBoost). The code provided here includes feature calculation scripts and the model training/inference pipeline.

## 📁 Code Structure
maize-phenology-nec/

├── features/              # Scripts for calculating features

│   ├── calculate_vegetation_indices.py

│   └── calculate_thermal_variables.py

│   └── Feature_concate.py.py

├── models/                # Machine learning model scripts

│   └── xgboost_phenology_model.py

│   └── xgboost_AARPD_Predicted.py 

│   └── DOY interpolation calculation.py 

├── requirements.txt       # Python dependencies

└── README.md

## Installation & Usage

### Prerequisites
*   Python 3.8+

### Basic Usage
The code is structured for clarity. You can run the feature calculation and model scripts individually as needed.

## Citation
If you use this code in your research, please cite our manuscript:

> Authors et al. "A High-Resolution (500-m) Dataset for Mapping Key Agronomic Growth Stages of Maize in Northeast China". Scientific Data (under review).

## License
This project is licensed under the MIT License.
