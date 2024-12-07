# **Brain Stroke Prediction Analysis**  
This project explores the *Brain Stroke Dataset* with a focus on preprocessing, exploratory data analysis, and predictive modeling to understand factors influencing stroke occurrences.  

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
Understanding factors behind strokes is crucial for early intervention and prevention. This project leverages Python's data science libraries to:  
- Explore patterns in data,  
- Identify correlations between features, and  
- Build a predictive model for stroke occurrences.  

---

## **Dataset**  
The dataset comprises 3,481 rows and 8 columns, detailing individual health metrics and stroke outcomes:  

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
```

### **Loading the Dataset**
```python
df = pd.read_csv("C:/Users/THIS PC/Desktop/brain_stroke1.csv")
print(df.head(10))
```

### **Dataset Shape**
```python
print("The number of rows:", df.shape[0])
print("The number of columns:", df.shape[1])
```
Output:  
```
The number of rows: 3481  
The number of columns: 8  
```

### **Dataset Information**
```python
print(df.info())
```
Output:  

| Feature              | Non-Null Count | Data Type |  
|----------------------|----------------|-----------|  
| `gender`            | 3,481          | Object    |  
| `age`               | 3,481          | Integer   |  
| `hypertension`      | 3,481          | Object    |  
| `heart_disease`     | 3,481          | Integer   |  
| `avg_glucose_level` | 3,481          | Float     |  
| `bmi`               | 3,481          | Float     |  
| `smoking_status`    | 3,481          | Object    |  
| `stroke`            | 3,481          | Integer   |  

### **Missing Values Check**
```python
print(df.isnull().sum())
```
Output:  
```
No missing values in the dataset.
```

---

## **Exploratory Data Analysis (EDA)**  

### **Descriptive Statistics**
```python
print(df.describe())
```
| Feature              | Count   | Mean    | Std Dev  | Min   | 25%   | 50%   | 75%   | Max   |  
|----------------------|---------|---------|----------|-------|-------|-------|-------|-------|  
| `age`               | 3481.0  | 49.07   | 18.89    | 10.0  | 34.0  | 50.0  | 64.0  | 82.0  |  
| `avg_glucose_level` | 3481.0  | 108.77  | 48.13    | 55.1  | 77.4  | 92.4  | 116.2 | 271.7 |  
| `bmi`               | 3481.0  | 29.81   | 6.22     | 14.1  | 25.3  | 29.1  | 33.5  | 48.9  |  

### **Correlation Heatmap**
```python
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of All Features', fontsize=18)
plt.show()
```
![Correlation Heatmap](path-to-image/correlation_heatmap.png)  

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
The analysis highlights key factors associated with strokes, such as age, BMI, and glucose levels. Logistic regression yielded preliminary insights, but improvements could be made using more advanced techniques.  

---
