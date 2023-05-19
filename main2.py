import pickle
import re
import warnings
from sklearn import svm
from os import path
import numpy as np
from nltk import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt, pyplot
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from textblob.tokenizers import word_tokenize

warnings.filterwarnings('ignore')

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def logistic_regression(X_train, Y_train, X_test, Y_test):
    print("------------Logistic Regression------------")
    if not path.isfile('LG_Model.sav'):
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        save_object(model, "LG_Model.sav")
    else:
        model = load_object("LG_Model.sav")
    Y_pred = model.predict(X_test)
    # confusion matrix
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
    print("Confusion matrix", confusion_mat)
    print(classification_report(Y_test, Y_pred))
    print("Score: ", model.score(X_test, Y_test))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu")
    pyplot.show()

def knn(X_train, Y_train, X_test, Y_test):
    print("-------------KNN-----------------------")
    if not path.isfile('KNN_Model.sav'):
        scaler = StandardScaler(with_mean=False)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, Y_train)
        save_object(knn, "KNN_Model.sav")
    else:
        knn = load_object("KNN_Model.sav")
    y_pred = knn.predict(X_test)

    print("Accuracy Score:", accuracy_score(Y_test, y_pred))
    print("Confusion matrix", confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, cmap="YlGnBu")
    pyplot.show()



def random_forest(X_train, Y_train, X_test, Y_test):
    print("------------Random Forest------------")
    if not path.isfile('RF_Model.sav'):
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=10)
        rf.fit(X_train, Y_train)
        save_object(rf, "RF_Model.sav")
    else:
        rf = load_object("RF_Model.sav")
    Y_pred = rf.predict(X_test)
    # confusion matrix
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
    print("Confusion matrix", confusion_mat)
    print(classification_report(Y_test, Y_pred))
    print("Score: ", rf.score(X_test, Y_test))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu")
    pyplot.show()


def decision_tree(X_train, Y_train, X_test, Y_test):
    print("-------------Decision Tree-----------------------")
    if not path.isfile('DT_Model.sav'):
        dt = DecisionTreeClassifier()
        dt.fit(X_train, Y_train)
        save_object(dt, "DT_Model.sav")
    else:
        dt = load_object("DT_Model.sav")
    y_pred = dt.predict(X_test)

    print("Accuracy Score:", accuracy_score(Y_test, y_pred))
    print("Confusion matrix", confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, cmap="YlGnBu")
    pyplot.show()

def preprocess_data(df):
    print("----------------Data Preprocessing-------------------")
    # Remove duplicates
    df.drop_duplicates(subset='Description', inplace=True)

    # Remove non-alphabetic characters
    df['Description'] = df['Description'].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))

    # Convert to lowercase
    df['Description'] = df['Description'].apply(lambda x: x.lower())

    # Tokenize
    df['Description'] = df['Description'].apply(word_tokenize)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    df['Description'] = df['Description'].apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    df['Description'] = df['Description'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Convert back to string
    df['Description'] = df['Description'].apply(lambda x: ' '.join(x))

    return df

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import SelectFromModel

def vectorize_data(train_df, test_df):
    print("------------------Data Vectorization------------------")
    # create a corpus of text
    train_corpus = train_df['Description'].tolist()
    test_corpus = test_df['Description'].tolist()

    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab on training data
    vectorizer.fit(train_corpus)

    # encode documents on training and testing data
    X_train = vectorizer.transform(train_corpus)
    Y_train = train_df['Rate']
    X_test = vectorizer.transform(test_corpus)
    Y_test = test_df['Rate']

    # scale the data
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # perform feature selection using logistic regression
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=10, random_state=10)
    selector = SelectFromModel(lr)
    selector.fit(X_train, Y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    # print the selected features
    feature_names = vectorizer.get_feature_names_out()
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"Selected Features: {selected_features}")

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('games-classification-dataset.csv')

    # Extract a subset of the data to speed up processing
    df = df.sample(frac=0.1, random_state=10)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)

    # Preprocess the training and testing datasets
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Vectorize the training and testing datasets
    X_train, Y_train, X_test, Y_test = vectorize_data(train_df, test_df)

    # Perform logistic regression on the training and testing datasets
    logistic_regression(X_train, Y_train, X_test, Y_test)

    # Perform KNN on the training and testing datasets
    knn(X_train, Y_train, X_test, Y_test)

    decision_tree(X_train, Y_train, X_test, Y_test)

    random_forest(X_train, Y_train, X_test, Y_test)