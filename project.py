
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

data=pd.read_csv("student_exam_scores.csv")
X=data[["hours_studied"]]
y=data["exam_score"]

model=LinearRegression()
model.fit(X,y)
predicted_scores=model.predict(X)
# Evaluate the model
mae=mean_absolute_error(y,predicted_scores)
mse=mean_squared_error(y,predicted_scores)
rmse=np.sqrt(mse)
#show the results
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)

new_hour=float(input("Enter hours studied to predict exam score: "))
new_prediction=model.predict([[new_hour]])
print("Predicted exam score for",new_hour,"hours studied is:",new_prediction)    