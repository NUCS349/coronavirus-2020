# Exploring Coronavirus data 

This repository contains some basic set up for exploration of the JHU data regarding
the Coronavirus pandemic of 2020.

## Setting up

Create a conda environment and activate it:

```
conda env create -f env.yml
conda activate coronavirus
```

## Get the data

```
git submodule update --init --recursive
```

The data is now at `COVID-19`. The data is updated daily so you can always keep pulling
new data in as it becomes available.

## What to do

This assignment is really open-ended and exploratory in nature. You can come up with
interesting visualizations of the data, run some machine learning experiments. Those
of you that are interested in pursuing further, I hope that this repository will be
a good starting point to incorporate other data sources (e.g. flight data, internet
traffic, and so on).

There is a very simple experiments set up in this repository to get started with.
One tries to predict the country with the most similar trajectory to a chosen country
using k-nearest neighbors. There are two variants of this experiment:

- `exp/knn_raw.py`: Uses the absolute case numbers to find the nearest neighbor 
  to each country by leaving the chosen country out of fitting. There are some 
  hyperparameters listed at the top of the file that are self-explanatory.
- `exp/knn_diff.py`: Uses the difference between day to day in case numbers to
  to find the most similar country to the chosen one. It has the same hyperparameters
  as before.
- `exp/knn_dist_diff.py`: This is time-series data so there may be delays between
  different countries that the other two experiments won't capture. In this 
  experiment, we take the difference between days, normalize it, and finally take
  the histogram of the data. This produces a distribution across the differences (the spikes or drops in cases from day to day). We use this as a feature vector
  to do k-nearest neighbor.

One option is to explore the hyperparameters of these scripts. You can also use these
scripts as a basis for performing other prediction tasks, such as trying to predict the
next days cases based on the previous days cases (e.g. fitting polynomial or exponential
models to time-series data).

For visualization, there is one script:

- `exp/viz.py`: plots the trajectories of each country in a single
  plot over time.

The given scripts are run from within the `exp` directory as follows:

```
cd exp
python knn_raw.py
python knn_diff.py
python viz.py
```

Finally, once you're done either visualizing the data or doing some machine learning on it
(or both), give a write-up (at least 1 page) of what you did and how it worked. This assignment will
not be graded harshly, at all. I encourage you to try it if you're interested!

**NOTE: Simply running the code that is already in this repository is not enough for credit.**

Ideally, one should expand significantly on the code in this repository either through
additional experiments on new tasks, thorough analysis and write-up of existing experiments,
or interesting visualizations of the data combined with machine learning, or some 
combination of all of these. Feel free to pull in additional data, and to use the data
sources included in this repo that are currently not in the experiments (the recovered csv, the death csv, the daily reports,
and the situation reports).

Thanks!
