"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_19-covid-Confirmed.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

plt.figure(figsize=(10, 10))
legend = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    if cases.sum() > MIN_CASES:
        plt.ylabel('# of Cases')
        plt.xlabel("Time")
        plt.plot(cases)
        legend.append(labels[0, 1])

plt.yscale('log')
plt.legend(legend)
plt.tight_layout()
plt.savefig('results/cases_by_country.png')