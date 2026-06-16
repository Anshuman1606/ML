import pandas as pd
df=pd.read_csv("student_placement_prediction_dataset_2026.csv")
print("sample rows")
print(df.head())

print("Dataset Shape:")
print(f"Rows:{df.shape[0]}, Columns:{df.shape[1]}")

print("Dataset Info:")
print(df.info())

print("Summary Stats:")
print(df.describe(include='all'))

print("Missing Values:")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
oeh=OneHotEncoder(drop='first',sparse_output=False)
oe=OrdinalEncoder(categories=[['Tier 1','Tier 2','Tier 3']])
df["college_tier"]=oe.fit_transform(df[["college_tier"]])
df["branch"]=oeh.fit_transform(df[["branch"]])
le=LabelEncoder()
df["volunteer_experience"]=le.fit_transform(df["volunteer_experience"])
df["placement_status"]=le.fit_transform(df["placement_status"])

df["hackathons_participated"]=le.fit_transform(df["hackathons_participated"])  

print("After encoding categorical variables")
print(df.head())

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features=df.columns.drop(["placement_status","gender"])#features
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])
X=df_scaled[features]#features
y=df_scaled["placement_status"]#target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)   
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test,y_pred)) 
print("Confusion Matrix:")
conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed","Placed"],yticklabels=["Not Placed","Placed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Skewness of Features:")
print(df[features].skew())
#print("Prediction for new student placement status:")
##try:
    ##new_student_data={
        ##"college_tier":int(input("Enter college tier (1, 2, or 3): ")),
    ##"communication_skill_score":float(input("Enter communication skill score (0-100): ")),
        ##"loical_reasoning_score":float(input("Enter logical reasoning score (0-100): ")),
        ##"attendance_percentage":float(input("Enter attendance percentage (0-100): ")),
        ##"internships_count":int(input("Enter number of internships completed: ")),
######new_student_scaled=scaler.transform(new_student_df)
    ##placement_prediction=model.predict(new_student_scaled)
    #result="Placed" if placement_prediction[0]==1 else "Not Placed"
    #print(f"Predicted placement status for the new student is: {result}")
#except Exception as e:
    #print(f"Error in input: {e}")




