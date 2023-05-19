import pandas as pd
import numpy as np
import pickle
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,BayesianRidge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve

def save_object(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load_object(filename):
    return pickle.load(open(filename, 'rb'))

df = pd.read_csv('games-regression-dataset.csv', sep=',', header=0, parse_dates=["Original Release Date","Current Version Release Date"], dayfirst=True)
print(df.shape)

# Missing Data
print(df.isna().sum()) #Subtitle, In-app Purchases, Languages
df['Subtitle'] = df['Subtitle'].fillna(' ')
df['In-app Purchases'] = df['In-app Purchases'].apply(lambda x: 0 if x != x else 1)
df['Languages'] = df['Languages'].fillna(' ')

# Feature Engineering
df['Price'] = df['Price'].apply(lambda x: 0 if x == 0 else 1)
df['Languages'] = df['Languages'].apply(lambda x: x.count(',') + 1)

df['Current Version Release Year']=pd.to_datetime(df['Current Version Release Date']).dt.year
df['Current Version Release Month']=pd.to_datetime(df['Current Version Release Date']).dt.month
df['Current Version Release Day']=pd.to_datetime(df['Current Version Release Date']).dt.day

df['Original Release Year']=pd.to_datetime(df['Original Release Date']).dt.year
df['Original Release Month']=pd.to_datetime(df['Original Release Date']).dt.month
df['Original Release Day']=pd.to_datetime(df['Original Release Date']).dt.day
df=df.drop(columns=['Current Version Release Date'])
df=df.drop(columns=['Original Release Date'])
################################################################################################
df['Description'] = df['Description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop = stopwords.words('english')
df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
st = PorterStemmer()
df['Description'] = df['Description'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['Description'] = df['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(df['Description'])
######################################################


#scaling
le = preprocessing.LabelEncoder()
df['Age Rating'] = le.fit_transform(df['Age Rating'])
df['Primary Genre'] = le.fit_transform(df['Primary Genre'])
df = pd.concat([df['Genres'].str.get_dummies(sep=','), df.drop('Genres', axis=1)], axis=1)
###############################################################################################
tfidf_vect = TfidfVectorizer(max_features=500, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
df['Description'] = le.fit_transform(df['Description'])
print(df['Description'])
##########################################################################
df.drop('URL', axis = 1, inplace = True)
df.drop('ID', axis = 1, inplace = True)
df.drop('Name', axis = 1, inplace = True)
df.drop('Subtitle', axis = 1, inplace = True)
df.drop('Icon URL', axis = 1, inplace = True)
#df.drop('Description', axis = 1, inplace = True)
df.drop('Developer', axis = 1, inplace = True)

corr = df.corr()
top_feature = corr.index[abs(corr['Average User Rating'])>0.04]
print(top_feature)
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
df=df[top_feature.values]


X = df.drop(['Average User Rating'], axis=1)
print(X)
Y = df['Average User Rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
scalingg= StandardScaler()
XR_train = pd.DataFrame(scalingg.fit_transform(X_train),columns=X.columns)
XR_test = pd.DataFrame(scalingg.transform(X_test),columns=X.columns)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
print("\nRidge Regression")
rd = Ridge()
rd.fit(XR_train,Y_train)
prediction = rd.predict(XR_test)
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))

#
#
print("\nlienear Regression")
LR = LinearRegression()
LR.fit(X_train,Y_train)
prediction = LR.predict(X_test)
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))
#
#
#
print("\nPolynomial Regression")
tran=PolynomialFeatures(2)
x1=tran.fit_transform(X_train)
cls = LinearRegression()
cls.fit(x1,Y_train)
prediction = cls.predict(tran.fit_transform(X_test))
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))



print("\nRandom Forest Regression")
FR = LinearRegression()
LR = RandomForestRegressor()
LR.fit(X_train,Y_train)
prediction = LR.predict(X_test)
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))



print("\nBayesianRidge Regression")
reg = BayesianRidge()
reg.fit(X_train,Y_train)
prediction= reg.predict(X_test)
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))



print("\nLasso Regression")
reg = Lasso()
reg.fit(X_train,Y_train)
prediction= reg.predict(X_test)
print("R-squared score: ",r2_score(Y_test,prediction))
print("MSE: ",mean_squared_error(Y_test,prediction))
