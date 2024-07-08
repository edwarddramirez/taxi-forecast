[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwarddramirez/taxi-forecast/HEAD) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/license/mit) ![Python](https://img.shields.io/badge/python-3.11.4-blue.svg) ![Repo Size](https://img.shields.io/github/repo-size/edwarddramirez/taxi-forecast) 

# taxi-forecast

## Introduction
Knowing where to go to find customers is the most important question for taxi drivers and hail-riding networks. If demand for taxis can be reliably predicted in real-time, taxi companies can dispatch drivers in a timely manner and drivers can optimize their route decision to maximize their earnings in a given day. This project aims to use rich trip-level data from NYC Taxi and Limousine Commission to construct time-series taxi rides data for 40,000 routes and forecast demand for rides. We will explore deep learning models for time series, such RNNs (LSTM), DeepAR, Transformers, and compare them with baseline statistical models, such as ARIMA and VAR.

## Installation
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==22.9.0`. This will install all packages needed to run the code on a CPU with `jupyter`. 

`Note:` This file has a lot of junk right now. Tailor it to the project's needs later down the line.

## Directory Structure
- `data`: Data Files
- `data_processing`: Notebooks for processing the data
- `models`: (Tentative) Modules of custom models used to perform fits
- `notebooks`: Notebook files performing fits and generating primary results
- `utils`: Custom modules or files 
- `scratch`: For unclean files used to develop code

Each directory contains an individual `README.md` file with more details of directory contents.

## Contributors
- [Edward Ramirez](https://github.com/edwarddramirez)
