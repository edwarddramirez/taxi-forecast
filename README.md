[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwarddramirez/taxi-forecast/HEAD) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/license/mit) ![Python](https://img.shields.io/badge/python-3.12.3-blue.svg) ![Repo Size](https://img.shields.io/github/repo-size/edwarddramirez/taxi-forecast) 

# taxi-forecast

## Introduction
Knowing where to go to find customers is the most important question for taxi drivers and hail-riding networks. If demand for taxis can be reliably predicted in real-time, taxi companies can dispatch drivers in a timely manner and drivers can optimize their route decision to maximize their earnings in a given day. This project aims to use rich trip-level data from NYC Taxi and Limousine Commission to construct time-series taxi rides data for 40,000 routes and forecast demand for rides. We will explore deep learning models for time series, such RNNs (LSTM), DeepAR, Transformers, and compare them with baseline statistical models, such as ARIMA and VAR.

## Installation

### Base Environment
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==22.9.0`. This will install all packages needed to run the data processing code and ARIMAX fitting notebooks with `jupyter` or `Binder`. 

### GPU Environments
The model training notebooks were built using `Google Colaboratory`. The `MLP`, `RNN`, and `LSTM` models are built using `pytorch=2.3.1` (i.e., the most updated version of `pytorch` on Google Colaboratory when we started this project). Therefore, the notebooks training these models should work out-of-the-box if you open them on Colab. 

On the other hand, our graph neural networks were built using the `torch-geometric-temporal` package. This package takes a long time to install and requires some patching due to incompatibility with our version of `pytorch`. We show how to install a permanent environment in `Google Drive` in this [Colab Notebook](https://github.com/edwarddramirez/taxi-forecast/blob/main/assets/colab/01_pytgt_test.ipynb). To install the package without a permanent environment, see this [Colab Notebook](https://github.com/edwarddramirez/taxi-forecast/blob/main/assets/colab/00_pytgt_test_no_permanent_env.ipynb) (not recommended). 

## Directory Structure
- `assets`: Additional assets unrelated to taxi data
- `data`: Taxi data directory
- `data_processing`: Notebooks for processing the taxi data
- `notebooks`: Notebook files summarizing the data, performing fits, and generating main results
- `utils`: Custom modules or files 
- `scratch`: For unclean files used to develop code

Each directory contains an individual `README.md` file with more details of directory contents.

## Contributors
- [Edward Ramirez](https://github.com/edwarddramirez)
- [Nazanin Komeilizadeh](https://github.com/NazThePhysicist)
- [Jade Ngoc Nguyen](https://github.com/jadenguyen)
- [Noah Gillespie](https://github.com/NoahGillespie)
- [Sriram Raghunath](https://github.com/sriramr30)
- [Li Meng](https://github.com/limeng-math)