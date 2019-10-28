## Description

BayesProcess is a python package for Physics informed Bayesian network inference using neural network surrogate model for matching process / variable / performance in solar cells.

## Installation

To install, just clone the following repository:
pip install -r requirements.txt
https://github.com/PV-Lab/BayesProcess.git

## Usage

run `surrogate_model.py` , with the given datasets to create the neural network surrogate for numerical PDE solver.
run `Bayes.py` with the saved surrogate model. This performs Bayesian network inference to map the process variable (Temperature) to material descriptors. 
The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `JV_surrogate.py`      | Script for training neural network JV surrogate model      |
| `Bayes.py`      | Script for Bayesian inference using MCMC    |


## Authors
"Danny" Zekun Ren and Felipe Oviedo
