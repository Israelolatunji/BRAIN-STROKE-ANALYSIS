# IMPORTING THE REQUIRED LIBRARIES 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pandas import DataFrame, Series

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

# Loading the brainstroke data set
df = pd.read_csv("C:/Users/THIS PC/Desktop/brain_stroke1.csv")

print(df.head(10))

   gender   age      hypertension       heart_disease    avg_glucose_level     bmi  \
0    Male    67   Not Hypertensive              1             228.69           36.6   
1    Male    80   Not Hypertensive              1             105.92           32.5   
2  Female    49   Not Hypertensive              0             171.23           34.4   
3  Female    79      Hypertensive               0             174.12           24.0   
4    Male    81   Not Hypertensive              0             186.21           29.0   
5    Male    74      Hypertensive               1             70.09            27.4   
6  Female    69   Not Hypertensive              0             94.39            22.8   
7  Female    81      Hypertensive               0             80.43            29.7   
8  Female    61   Not Hypertensive              1             120.46           36.8   
9  Female    54   Not Hypertensive              0             104.51           27.3   

      smoking_status       stroke  
0    formerly smoked          1  
1     never smoked            1  
2           smokes            1  
3     never smoked            1  
4    formerly smoked          1  
5     never smoked            1  
6     never smoked            1  
7     never smoked            1  
8           smokes            1  
9           smokes            1  
