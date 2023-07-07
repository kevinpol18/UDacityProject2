# Enhanced Disaster Response Web Application
## Setup and Requirements

This repository utilizes Python 3 and includes the following necessary packages: json, pandas, nltk, flask, sklearn, sqlalchemy, numpy, re, and pickle.

## Comprehensive Project Description

The purpose of this repository is to provide a robust web application that emergency responders can use during natural disasters such as earthquakes or hurricanes. The app classifies incoming disaster messages into distinct categories, facilitating the process of directing messages to the appropriate relief organizations.

The application employs a machine learning (ML) model to categorize new messages. Additionally, the repository contains the code responsible for training the model and preparing any new datasets for model training purposes.

## Detailed File Overview
process_data.py: This script takes csv files containing message data and message categories (labels) as input and generates an SQLite database with a cleaned and consolidated version of the data.
train_classifier.py: This script accepts the SQLite database generated by process_data.py as input and uses the data to train and fine-tune an ML model for message categorization. The result is a pickle file containing the fitted model. Test evaluation metrics are printed during the training process.
ETL Pipeline Preparation.ipynb: This Jupyter notebook contains the code and analysis used in the development of process_data.py. Essentially, process_data.py automates this notebook.
ML Pipeline Preparation.ipynb: This Jupyter notebook contains the code and analysis used in the development of train_classifier.py. It specifically includes the analysis used to fine-tune the ML model and determine the optimal algorithm. train_classifier.py automates the model fitting process outlined in this notebook.
data: This folder stores sample messages and categories datasets in csv format.
app: This folder houses all necessary files to run and display the web application.
## Step-by-Step Execution Guide
Run `process_data.py`
Store the data folder in the current working directory and place process_data.py inside the data folder.
In the current working directory, execute the following command: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
Run `train_classifier.py`

## Imbalance Warning
Please note that the datasets in this repository are highly imbalanced, with several message categories containing very few positive examples. In some instances, the proportion of positive examples is less than 5%, or even less than 1%. Consequently, although the classifier's accuracy may be high (as it generally predicts that messages do not belong to these categories), the classifier's recall (i.e., the proportion of positive examples correctly labeled) tends to be quite low. Therefore, exercise caution when relying on the results of this app for decision-making purposes.

## License, Authors, and Acknowledgments
This web application was developed as part of the Udacity Data Scientist Nanodegree program. Udacity provided code templates and data. The original data was sourced by Udacity from Figure Eight.
Create a folder named 'models' in the current working directory and store train_classifier.py inside it.
In the current working directory, execute the following command: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
Launch the web application
Store the app folder in the current working directory.
Run the following command inside the app directory: python run.py
