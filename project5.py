import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

df=pd.read_csv("train.csv")
##print(df.head())
df.drop(columns=["PassengerId","Name","Ticket","Cabin"],inplace=True)
X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=["Survived"]),df["Survived"],test_size=0.2,random_state=42)
##print(X_train.head())
##print(y_train.sample(6))

##imputation Transformer
trf1=ColumnTransformer([
    ("impute_age",SimpleImputer(),[2]),
    ("impute_embarked",SimpleImputer(strategy="most_frequent"),[6])
],remainder="passthrough")

##one-hot encoding 
trf2=ColumnTransformer([
    ("ohe_sex_embarked",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),[1,6])
],remainder="passthrough")
##scaling
trf3=ColumnTransformer([
    ("scale",MinMaxScaler(),slice(0,10))])
##feature selection
trf4=SelectKBest(score_func=chi2,k=8)
##model
trf5=DecisionTreeClassifier()

##pipeline
pipe=Pipeline([
    ("trf1",trf1),
    ("trf2",trf2),
    ("trf3",trf3),
    ("trf4",trf4),
    ("trf5",trf5)
])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print("Predictions:", y_pred)
from sklearn.utils import estimator_html_repr

 #Generate the HTML structure of your pipeline
html_data = estimator_html_repr(pipe)

 ##Save it to an HTML file
with open("pipeline_diagram.html", "w", encoding="utf-8") as f:
   f.write(html_data)

print("Pipeline diagram saved successfully! Open 'pipeline_diagram.html' in your browser.")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import cross_val_score
cross_val_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", np.mean(cross_val_scores))

params= {
    "trf5__max_depth":[1,2,3,5,7,None]}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(pipe,param_grid=params,cv=5,scoring="accuracy")
grid_search.fit(X_train,y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)