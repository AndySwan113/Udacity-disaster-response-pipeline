# Udacity Disaster Response Pipeline Project

https://github.com/AndySwan113/Udacity-disaster-response-pipeline

## Introduction:
For this project we were given a data set containing real messages that were sent during disaster event and had to create a machine learning pipeline to categorize these events to their relevent groups.
This is in the form of a web app where an emergency worker can input a new message and get classification results in several categories.

## File Descriptions:
Data:
disaster_messages.csv - CSV of messages sent during disaster events

disaster_categories.csv - CSV of the categories of the messages

ETL Pipeline Preparation.ipynb - Notebook that was used when creating the ETL pipeline

process_data.py - ETL pipeline above that was used to store data in SQLite database
DisasterResponse.db - cleaned datafame stored in SQlite database

Model:
ML Pipeline Preparation.ipynb - Notebook used when investingating different Machine Learning pipelines

train_classifier.py - Using the most accurate Machine Learning pipeline used to train model

model.pkl - pickle file contains trained model

App:
run.py - python script to run web appl.

## Running Web App
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
        
     - To run ML pipeline that trains classifier and saves
        `python model/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

## Libraries Used:
import pandas as pd

import numpy as np

from sqlalchemy import create_engine

import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re

import numpy as np

import pandas as pd

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier


## Licenses
Figure Eight - Provided the datasets 
