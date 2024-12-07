# IMPORTING THE REQUIRED LIBRARIES 

```
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
```
   gender  age      hypertension  heart_disease  avg_glucose_level   bmi  \
0    Male   67  Not Hypertensive              1             228.69  36.6   
1    Male   80  Not Hypertensive              1             105.92  32.5   
2  Female   49  Not Hypertensive              0             171.23  34.4   
3  Female   79      Hypertensive              0             174.12  24.0   
4    Male   81  Not Hypertensive              0             186.21  29.0   
5    Male   74      Hypertensive              1              70.09  27.4   
6  Female   69  Not Hypertensive              0              94.39  22.8   
7  Female   81      Hypertensive              0              80.43  29.7   
8  Female   61  Not Hypertensive              1             120.46  36.8   
9  Female   54  Not Hypertensive              0             104.51  27.3   

    smoking_status  stroke  
0  formerly smoked       1  
1     never smoked       1  
2           smokes       1  
3     never smoked       1  
4  formerly smoked       1  
5     never smoked       1  
6     never smoked       1  
7     never smoked       1  
8           smokes       1  
9           smokes       1  
 


# Part A: data preprocessing
print("The number of rows:", df.shape[0])

print("The number of columns:", df.shape[1])

The number of rows: 3481

The number of columns: 8

# Checking information about the data set
print(df.info())

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 3481 entries, 0 to 3480

Data columns (total 8 columns):
 ###      Column                  Non-Null Count       Dtype  
---       ------                  --------------      -----  
 0        gender                  3481 non-null       object
 
 1        age                     3481 non-null       int64  
 2        hypertension            3481 non-null       object 
 3        heart_disease           3481 non-null       int64  
 4        avg_glucose_level       3481 non-null       float64
 5        bmi                     3481 non-null       float64
 6        smoking_status          3481 non-null       object 
 7        stroke                  3481 non-null       int64  
dtypes: float64(2),  int64(3),  object(3)
memory usage: 217.7+ KB
None

# Checking for Missing values in columns
gender               0

age                  0

hypertension         0

heart_disease        0

avg_glucose_level    0

bmi                  0

smoking_status       0

stroke               0

dtype: int64

As seen above the data set have no mising values in any of the columns

# Part B: EXPLORATORY DATA ANALYSIS 

## Descriptive Analysis of the quantitative variables
                   count        mean        std    min    25%    50%     75%  \ 
age                3481.0   49.067509  18.894255  10.00  34.00  50.00   64.00   
heart_disease      3481.0    0.065211   0.246934   0.00   0.00   0.00    0.00   
avg_glucose_level  3481.0  108.769155  48.133475  55.12  77.45  92.49  116.25   
bmi                3481.0   29.808159   6.228310  14.10  25.30  29.10   33.50   
stroke             3481.0    0.057742   0.233288   0.00   0.00   0.00    0.00   

                      max  
age                 82.00  
heart_disease        1.00  
avg_glucose_level  271.74  
bmi                 48.90  
stroke               1.00  

# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, data=df)
plt.title('Correlation Heatmap of All Features', fontsize=18)
plt.show()




