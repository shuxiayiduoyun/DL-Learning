#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：openmml1.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/11/12 21:49 
'''
import openml
from sklearn import ensemble
from openml import tasks, runs

# List all datasets and their properties
# openml.datasets.list_datasets(output_format="dataframe")

# Get dataset by ID
dataset = openml.datasets.get_dataset(61)

# Get dataset by name
dataset = openml.datasets.get_dataset('Fashion-MNIST')

# Get the data itself as a dataframe (or otherwise)
X, y, _, _ = dataset.get_data(dataset_format="dataframe")
print(type(X))

# Build any model you like
clf = ensemble.RandomForestClassifier()

# Download any OpenML task
task = tasks.get_task(3954)

# Run and evaluate your model on the task
run = runs.run_model_on_task(clf, task)
