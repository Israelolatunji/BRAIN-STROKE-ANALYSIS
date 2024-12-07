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
### Dataset Information  
- **Shape:** 3,481 rows × 8 columns  
- **Memory Usage:** ~217 KB  
- **Missing Values:** None  

```python  
print(df.info())  
```

| **Feature**         | **Non-Null Count** | **Data Type** |  
|----------------------|-------------------|---------------|  
| `gender`            | 3,481             | Object        |  
| `age`               | 3,481             | Integer       |  
| `hypertension`      | 3,481             | Object        |  
| `heart_disease`     | 3,481             | Integer       |  
| `avg_glucose_level` | 3,481             | Float         |  
| `bmi`               | 3,481             | Float         |  
| `smoking_status`    | 3,481             | Object        |  
| `stroke`            | 3,481             | Integer       |  

---

## **Exploratory Data Analysis (EDA)**  

### Descriptive Statistics  
```python  
df.describe()  
```  
| Feature              | Count   | Mean    | Std Dev  | Min   | 25%   | 50%   | 75%   | Max   |  
|----------------------|---------|---------|----------|-------|-------|-------|-------|-------|  
| `age`               | 3481.0  | 49.07   | 18.89    | 10.0  | 34.0  | 50.0  | 64.0  | 82.0  |  
| `avg_glucose_level` | 3481.0  | 108.77  | 48.13    | 55.1  | 77.4  | 92.4  | 116.2 | 271.7 |  
| `bmi`               | 3481.0  | 29.81   | 6.22     | 14.1  | 25.3  | 29.1  | 33.5  | 48.9  |  

### Correlation Heatmap  
The heatmap reveals relationships between numerical variables.  

```python  
plt.figure(figsize=(12,8))  
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)  
plt.title('Correlation Heatmap of All Features', fontsize=18)  
plt.show()  
```  

![Correlation Heatmap](path-to-image/correlation_heatmap.png)  

---

## **Model Building**  
### Logistic Regression  
A logistic regression model is implemented to predict stroke occurrences.  
- **Train/Test Split:** 80/20  
- **Metrics Evaluated:**  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  

```python  
model = LogisticRegression()  
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)  
```  

---

## **Conclusion**  
The project provides insights into factors like age, glucose levels, and BMI influencing stroke risks. Further model improvements could include advanced algorithms and hyperparameter tuning.  

---

## **Future Work**  
- Incorporating additional features,  
- Exploring ensemble methods, and  
- Optimizing for imbalanced datasets.  

---

Feel free to contribute, suggest improvements, or fork this repository!  

--- 
