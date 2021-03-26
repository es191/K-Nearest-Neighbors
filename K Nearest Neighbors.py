#Import the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Reading our file
df = pd.read_csv('KNN_Project_Data')
df.head()
df.describe()
df.info()

#Creating a pairplot with the hue is Target class
sns.pairplot(df, hue ='TARGET CLASS')

#Creating a StandardScaler() object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaler_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaler_features, columns = df.columns[:-1])
df_feat.head()

#Spliting our data to train and test data
from sklearn.model_selection import train_test_split
X = df_feat
y= df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Creating a KNN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors=1)

#Fitting our model
knn.fit (X_train, y_train)

#Prediction of our model and first evaluation
prediction = knn.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report (y_test, prediction))
print (confusion_matrix (y_test, prediction))

#Creating a range with different values for K
error_rate =[]
for i in range (1,60):
    knn = KNeighborsClassifier (n_neighbors=i)
    knn.fit (X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_rate.append (np.mean(prediction_i != y_test))

#Plot for our error_rate
plt.figure (figsize = (13,7))
plt.plot (range(1,60), error_rate,  marker = 'o')
plt.title ('Comparing with different K value')
plt.xlabel('K')
plt.ylabel('Error rate')

#Choosing the best value for K 
knn = KNeighborsClassifier (n_neighbors=55)
knn.fit(X_train, y_train)
prediction = knn.predict (X_test)

#The final evaluation of our model
print(classification_report (y_test, prediction))
print (confusion_matrix (y_test, prediction))