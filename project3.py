import pandas as pd
df=pd.read_csv("student_performance_dataset.csv")
print("sample rows")
print(df.head())

print("Dataset shape")
print(f"Rows:{df.shape[0]}, Columns:{df.shape[1]}")

print("Dataset info")
print(df.info())

print("Summary stats")
print(df.describe(include='all'))

print("Missing values")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Internet_Access_at_Home"]=le.fit_transform(df["Internet_Access_at_Home"])
df["Pass_Fail"]=le.fit_transform(df["Pass_Fail"])

print("After encoding categorical variables")
print(df.head())

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features=["Study_Hours_per_Week","Internet_Access_at_Home","Final_Exam_Score","Attendance_Rate"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])

X=df_scaled[features]#features
y=df_scaled["Pass_Fail"]#target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test,y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
confusion_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("PREDICT YOUR RESULT")
try:
    study_hours_per_week=float(input("Enter hours studied per week: "))
    internet_access_at_home=int(input("Do you have internet access at home? (1 for Yes, 0 for No): "))
    final_exam_score=float(input("Enter your final exam score: "))
    attendance_rate=float(input("Enter your attendance rate (0-100): "))
    user_input_df=pd.DataFrame({'Study_Hours_per_Week': [study_hours_per_week], 
                                'Internet_Access_at_Home': [internet_access_at_home], 
                                'Final_Exam_Score': [final_exam_score], 
                                'Attendance_Rate': [attendance_rate]})
    user_input_scaled=scaler.transform(user_input_df)
    prediction=model.predict(user_input_scaled)[0]
    result="Pass" if prediction==1 else "Fail"
    print(f"Based on the input, you are predicted to: {result}")
except Exception as e:
    print("Error in input:", e)