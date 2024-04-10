# Classification Techniques on Adult Dataset

This repository contains the implementation and analysis of various classification techniques applied to the Adult dataset, which is a well-known dataset in machine learning for income prediction. The primary goal is to explore the effectiveness of different classifiers in predicting whether an individual's income exceeds $50K/year based on census data.

## Techniques Implemented
K-Nearest Neighbors (KNN): A simple, instance-based learning algorithm that classifies a new instance based on the majority label of its k nearest neighbors.
Random Forest: An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.
Logistic Regression: A statistical model that uses a logistic function to model a binary dependent variable.
Gaussian Naive Bayes: A probabilistic classifier based on applying Bayes' theorem with the assumption of independence between every pair of features.
Linear Support Vector Classification (Linear SVC): A linear version of the Support Vector Machine algorithm used for binary classification tasks.
XGBoost: An implementation of gradient-boosted decision trees designed for speed and performance.

## Data Preprocessing
Two approaches were used to clean the dataset:

Approach 1: Replacing missing values with NaN and dropping rows with NaN values.
Approach 2: Imputing missing values with the mode of each column.

## Classification Results
The classifiers were evaluated on their training and testing accuracy, and their error rates were compared with classical classifiers reported in the adult.names file.

## Downsampling Analysis
The impact of training data volume on the classifier's error rate was analyzed by downsampling the training dataset to different percentages (50%, 60%, 70%, 80%, and 90%) and recording the mean and standard deviation of the error rates.

## Improvements and Analysis
The performance of each classifier was analyzed, and suggestions for improvements were provided. XGBoost was proposed as a solution to beat all the classic classifiers reported in the adult.names file.

## Usage
Clone the repository to your local machine.
Open the Jupyter Notebook Classification_Techniques_Adult_Dataset.ipynb.
Run the cells to observe the results of different classification techniques on the Adult dataset.

## Dependencies
Python 3
NumPy
Pandas
scikit-learn
Matplotlib
