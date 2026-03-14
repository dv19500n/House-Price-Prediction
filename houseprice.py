import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


dataset=pd.read_excel("HousePricePrediction.xlsx")
#  Data Preprocessing
obj=(dataset.dtypes=='object')
objects=list(obj[obj].index)
print("Categorical variables: ",len(objects))
int_num=(dataset.dtypes=='int')
numbers=list(int_num[int_num].index)
print("Integer variables: ",len(numbers))
float_num=(dataset.dtypes=='float')
floats=list(float_num[float_num].index)
print("Float variables: ",len(floats))
# Exploratory Data Analysis 
# (HEATMAP)
numerical_dataset=dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(),cmap="inferno", fmt=".2f", linewidths=2,annot=True)
plt.show()

# (BARPLOT)
unique_values=[]
for column in objects:
    unique_values.append(dataset[column].unique().size)
plt.figure(figsize=(10,6))
plt.title("No. Unique values of Catigorical Features")
plt.xticks(rotation=90)
sns.barplot(x=objects, y=unique_values)

plt.figure(figsize=(18,36))
plt.title("Categorical Features: Distribution")
plt.xticks(rotation=90)
index=1
for column in objects:
    y=dataset[column].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index),y=y)
    index +=1
# data cleaning
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice']=dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset= dataset.dropna()
new_dataset.isnull().sum()
# labeling categorical features
s= (new_dataset.dtypes=='object')
object_columns=list(s[s].index)
print("Categorical variables:")
print(object_columns)
print("No. of. categorical features: ", len(object_columns))
OH_encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols=pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_columns]))
OH_cols.index=new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final=new_dataset.drop(object_columns, axis=1)
df_final=pd.concat([df_final,OH_cols], axis=1)
numerical_cols=df_final.select_dtypes(include=['int64','float64']).columns
if 'SalePrice' in numerical_cols:
    numerical_cols=numerical_cols.drop('SalePrice')
scaler=StandardScaler()
df_final[numerical_cols]=scaler.fit_transform(df_final[numerical_cols])
# splitting dataset into training and testing 
X=df_final.drop(['SalePrice'], axis=1)
Y=df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y, train_size=0.8, test_size=0.2, random_state=0)
# SVM - Support vector machine
model_SVR=svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred_SVR= model_SVR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_SVR))
# Random Forest Regression
model_RFR=RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_RFR=model_RFR.predict(X_valid)
mean_absolute_percentage_error(Y_valid, Y_pred_RFR)
# Linear Regression
model_LR=LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_LR=model_LR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_LR))
# model comparison
models=['SVR', 'Random Forest', 'Linear Regression']
PE_scores=[mean_absolute_percentage_error(Y_valid, Y_pred_SVR), 
        mean_absolute_percentage_error(Y_valid, Y_pred_RFR),
        mean_absolute_percentage_error(Y_valid, Y_pred_LR)]
plt.figure(figsize=(10,6))
plt.bar(models, PE_scores)
plt.title('Model Comparison - Percent Error Score')
plt.ylabel('Percent Error')
plt.xticks(rotation=45)
plt.show()
