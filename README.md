# Tweets Sentiment Analysis

This project focuses on performing sentiment analysis on tweets using machine learning and natural language processing techniques. The primary goal is to classify tweets into positive, negative, or neutral sentiments.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction

The aim of this project is to analyze the sentiment of tweets. Sentiment analysis is a crucial task in natural language processing with applications in social media monitoring, customer feedback analysis, and more.

## Dataset
- Source of the dataset : Hugging Face emotion dataset
- Number of samples: 16000(trainig), 2000(validation), 2000(testing)

## Requirements

List all the dependencies required to run the notebook:
- Python 3.x
- PyTorch
- Transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- umap learn

## Installation

Provide instructions for installing the necessary packages:
```bash

pip install transformers
pip install datasets
pip install huggingface_hub
pip install umap-learn[plot]
pip install umap-learn
pip install torch
```

## Usage

Instructions on how to use the notebook:
1. Clone the repository.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Model Training

Outline the model training process:
- Model architecture used : Distil-Bert, LogisticRegression

## Evaluation
- Metrics (accuracy, F1 score)

## Results

### Distil-Bert
  - Accuracy: 0.932949
  - F1 score: 0.933000
  - 
### LogisticRegression
  - Accuracy: 0.634
