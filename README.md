
---

# **Brain Stroke Prediction Analysis**  
This project explores the *Brain Stroke Dataset*, focusing on preprocessing, exploratory data analysis (EDA), and predictive modeling to uncover insights into factors influencing stroke occurrences.  

---

## **Table of Contents**  
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)  
5. [Model Building](#model-building)  
6. [Conclusion](#conclusion)  

---

## **Overview**  
Stroke is one of the leading causes of death worldwide.

Stroke occurs due to a decrease in oxygen to the brain. This can be due to a bleed or blockage in the brain’s blood supply. Immediate emergency treatment may help prevent life threatening consequences.

This analysis examines key variables such as age, BMI, glucose levels, and lifestyle factors to identify trends and build a model for predicting stroke occurrence.  

---

### **Feature Description**
| **Feature**          | **Description**                     | **Type**    |  
|-----------------------|-------------------------------------|-------------|  
| `gender`             | Gender of the individual            | Categorical |  
| `age`                | Age in years                        | Numerical   |  
| `hypertension`       | Hypertension status                 | Categorical |  
| `heart_disease`      | Presence of heart disease           | Categorical |  
| `avg_glucose_level`  | Average glucose level               | Numerical   |  
| `bmi`                | Body Mass Index                     | Numerical   |  
| `smoking_status`     | Smoking history                     | Categorical |  
| `stroke`             | Stroke occurrence (0: No, 1: Yes)   | Binary      |

--------------

## **Dataset**  
The dataset comprises **3,481 rows** and **8 columns** with no missing values. Below is a sample of the dataset:  

| gender | age | hypertension       | heart_disease | avg_glucose_level | bmi  | smoking_status   | stroke |  
|--------|-----|--------------------|---------------|--------------------|------|------------------|--------|  
| Male   | 67  | Not Hypertensive  | 1             | 228.69             | 36.6 | formerly smoked  | 1      |  
| Male   | 80  | Not Hypertensive  | 1             | 105.92             | 32.5 | never smoked     | 1      |  
| Female | 49  | Not Hypertensive  | 0             | 171.23             | 34.4 | smokes           | 1      |  
| Female | 79  | Hypertensive      | 0             | 174.12             | 24.0 | never smoked     | 1      |  
| Male   | 81  | Not Hypertensive  | 0             | 186.21             | 29.0 | formerly smoked  | 1      |  
| Male   | 74  | Hypertensive      | 1             | 70.09              | 27.4 | never smoked     | 1      |  
| Female | 69  | Not Hypertensive  | 0             | 94.39              | 22.8 | never smoked     | 1      |  
| Female | 81  | Hypertensive      | 0             | 80.43              | 29.7 | never smoked     | 1      |  
| Female | 61  | Not Hypertensive  | 1             | 120.46             | 36.8 | smokes           | 1      |  
| Female | 54  | Not Hypertensive  | 0             | 104.51             | 27.3 | smokes           | 1      |  
  

---

## **Data Preprocessing**  

### **Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### **Loading the Dataset**
```python
df = pd.read_csv("C:/Users/THIS PC/Desktop/brain_stroke1.csv")
print(df.head(10))

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


```

### **Dataset Shape**
```python
print("The number of rows:", df.shape[0])
print("The number of columns:", df.shape[1])

The number of rows is: 3481
The number of columns is: 8

```

## Checking for missing values
```python
missing_values = df.isnull().sum()

# Print the missing values for each column
print(missing_values)

# Check if there are any missing values in the entire dataset
if missing_values.sum() == 0:
    print("No missing values in the dataset.")
else:
    print("There are missing values in the dataset.")

```
| Feature             | Missing Values |
|---------------------|----------------|
| gender              | 0              |
| age                 | 0              |
| hypertension        | 0              |
| heart_disease       | 0              |
| avg_glucose_level   | 0              |
| bmi                 | 0              |
| smoking_status      | 0              |
| stroke              | 0              |

No missing values in the dataset.
```python

```

---

## **Exploratory Data Analysis (EDA)**  

### **Descriptive Statistics**
```python
print(df.describe().T)
```

| Metric               | age     | heart_disease | avg_glucose_level | bmi     | stroke   |
|----------------------|---------|---------------|-------------------|---------|----------|
| count               | 3481.0  | 3481.0        | 3481.0            | 3481.0  | 3481.0   |
| mean                | 49.0675 | 0.0652        | 108.7692          | 29.8082 | 0.0577   |
| std                 | 18.8943 | 0.2469        | 48.1335           | 6.2283  | 0.2333   |
| min                 | 10.00   | 0.00          | 55.12             | 14.10   | 0.00     |
| 25%                 | 34.00   | 0.00          | 77.45             | 25.30   | 0.00     |
| 50% (median)        | 50.00   | 0.00          | 92.49             | 29.10   | 0.00     |
| 75%                 | 64.00   | 0.00          | 116.25            | 33.50   | 0.00     |
| max                 | 82.00   | 1.00          | 271.74            | 48.90   | 1.00     |

'''

### **Correlation Heatmap**
```python
# Heatmap Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of All Features', fontsize=18)
plt.show()

```
![correlation_heatmap.png)  

---

## **Model Building**  

### **Train-Test Split**
```python
X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **Logistic Regression**
```python
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### **Model Evaluation**
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## **Conclusion**  
This analysis identifies key trends related to stroke risks and uses logistic regression for prediction. Future work will focus on improving the model and exploring advanced algorithms.  

---

Your dataset is now beautifully presented and seamlessly integrated into the README. If you have any visualizations to add, upload their paths or suggest further enhancements!
