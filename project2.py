import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
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
r2=r2_score(y,predicted_scores)
#show the results
print("Mean Absolute Error:",round(mae,2))
print("Mean Squared Error:",round(mse,2))
print("Root Mean Squared Error:",round(rmse,2))
print("R-squared score (model accuracy):",round(r2,2))
# Plotting the data and the regression line
plt.figure(figsize=(10,6))
plt.hist(data["exam_score"], bins=20, alpha=0.5, label="Hours Studied",color='blue',edgecolor='black')
plt.title("Distribution of scores")
plt.xlabel("Scores")
plt.ylabel("no.of students")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(X,y, label="actual scores",color='blue')
plt.plot(X, predicted_scores, label="predicted scores", color='red')
plt.title("model predicted vs actual scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

new_hours=float(input("Enter hours studied to predict exam score: "))
new_prediction=model.predict([[new_hours]])
print("Predicted exam score for",new_hours,"hours studied is:",(new_prediction))