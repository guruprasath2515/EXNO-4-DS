
# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method
# NAME: Guru prasath R
# REG NO: 212223040053
# CODING AND OUTPUT:
```py
# Feature Scaling
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('bmi.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/afb2e2fb-e46b-4c87-b200-0bf4056e59f9)

```py
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/7ee17ee2-6079-4f40-b825-2a708b2ac36b)

```py
df.dropna()
```
![image](https://github.com/user-attachments/assets/556d5d59-aadf-43ff-8c58-ab4d7a3e541b)

```py
 max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
 max_vals
```
![image](https://github.com/user-attachments/assets/ee68fea0-ff47-4390-b8b9-96d1fb0ec425)

```py
# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv('bmi.csv')
df1.head()
```
![image](https://github.com/user-attachments/assets/40f450a6-c34b-4a64-8046-b4fc7082a440)


```py
 sc=StandardScaler()
 df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
 df1.head(10)
```

```py
# MIN-MAX SCALING:
 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)
```
![image](https://github.com/user-attachments/assets/4bb850aa-0fa2-40a5-80ba-5cf4b8709b9d)

```py
# MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv('bmi.csv')
df3.head()
```
![image](https://github.com/user-attachments/assets/f9d714b5-8fbc-4124-8428-83b216ff579a)

```py
 df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
 df3
```
![image](https://github.com/user-attachments/assets/e377a67c-0303-42d3-b512-8c07bc41992e)

```py
# ROBUST SCALING:
 from sklearn.preprocessing import RobustScaler
 scaler = RobustScaler()
df4=pd.read_csv("bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/d3139cfb-045a-422a-8de5-69f88cc99820)

```py
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/30cd64b6-1822-4a58-b97b-0043e4d6b79e)

```py
# FEATURE SELECTION:
import pandas as pd
df=pd.read_csv("income.csv")
df.info()
```
![image](https://github.com/user-attachments/assets/92f7ef26-42af-45fa-bcc0-f77df1ee6385)

```py
df.head()
```
![image](https://github.com/user-attachments/assets/0d8e8e03-ca3b-4a8f-8b65-df5827e0dfd9)

```py
 df_null_sum=df.isnull().sum()
 df_null_sum
```
![image](https://github.com/user-attachments/assets/5af1eea9-9090-4ec0-8799-da83beb84cfd)

```py
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/1ce4d7cf-0fa1-4b90-8009-44c9709fa6e8)

```py
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/6864c0ce-634b-4d69-aacb-dbe7c9b729d9)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42))
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/6d8969c8-1ae3-4a28-966d-5fb96614a187)

```py
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/a9567cb7-418f-4a51-a650-1dc9f7898a49)

```py
# FILTER METHOD:
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/2814a605-8eeb-495a-bc05-d08043c29c7b)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8dd94fc7-eaab-4b20-85b9-f4cb4db423d8)

```py
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/24a6b73b-5aab-4e96-8eeb-64b761866018)

```py
# Model
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```
![image](https://github.com/user-attachments/assets/041c503c-a725-4efe-8697-7e12043ec6e5)

```py
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/479ee52b-cfca-4857-bf00-1b3344964a82)

```py
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/c94d94ab-663c-4fda-9cda-0bbeb8e0811b)

```py
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/b919df21-71f9-4cc9-9572-7c5ab1fe32e7)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/fea55a6c-82b3-4e76-9d8f-4eb16e537943)

```py
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8005ef40-376b-4fba-8097-e83157d28076)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/83f1eb16-d29a-4d27-90ad-986509327b08)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
