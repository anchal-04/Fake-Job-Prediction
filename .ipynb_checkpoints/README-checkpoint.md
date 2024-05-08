**Project Overview**

With the rise of online job portals and recruitment methods, the issue of fake job postings has become a significant issue. Fake job postings not only waste the time of job seekers but also pose a serious threat to their personal information like address, bank account, SSN etc. In this project, we aim to analyze job postings and distinguish between true and fake jobs. 

Data Source

The dataset used for this project is from Kaggle - https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data.
This dataset contains 18K job postings out of which about 800 are fake. 

Problem Statement

The aim of the project is to classify between real and fake jobs. The data used in this project has both text and numeric data and two models are used to deal with these two kinds of data. The result will be the combination of both the results.

Steps involved -
1. Data Collection
2. Data exploration and pre-processing
3. Modeling
4. Evaluating

Data Collection

The data is collected from Kaggle which has 17,880 observations and 18 features. Many of the fields have null values and are in different language thus those values are ignored.
 After analysis 10593 observations and 20 features are used.


Data Exploration and pre-processing

Different steps used in exploration are -
1. Correlation Matrix- A correlation matrix is a useful statistical technique to evaluate the relationship between two variables in a dataset.

Evaluating

While evaluating different models we will consider two metrics in mind-
1. Accuracy
 Accuracy = True Positive + True Negative /(True Positive + False Positive + True Negative + False Negative)

 This produces ration of all the correctly categorized jobs to all the jobs.

2. F1-score

F1 = True Positive/(True Positive + 1/2(False Positive + False Negative))

F1 score is a measure of model's accuracy on a dataset
