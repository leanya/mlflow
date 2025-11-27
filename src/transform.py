import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

def ingest_dataset(type):

    if type=="train":
        df_raw = pd.read_csv("./data/train.csv")
    else:
        df_raw = pd.read_csv("./data/test.csv")
    
    return df_raw

def data_cleaning_general(df):

    # Extract first char for Cabin to create col Cabin_Type
    df["Cabin_Type"] = df["Cabin"].str[0].replace("", pd.NA)

    # Categorical to Numeric
    df["Sex"] = df["Sex"].map({"female":1, "male":0})
    df["Embarked"] = df["Embarked"].map({"Q":2, "S":1, "C":0, np.nan:-1})

    # Drop Columns 
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    return df

def data_cleaning_train(df):

    # For train datasets only 
    # Impute Embarked with mode
    imputer_embarked = SimpleImputer(strategy="most_frequent")
    df["Embarked"] = imputer_embarked.fit_transform(df[["Embarked"]])[:, 0]
    print("Learned most frequent value for Embarked:", imputer_embarked.statistics_)

    # Impute Age with Median
    imputer_age = SimpleImputer(strategy="median")
    df["Age"] = imputer_age.fit_transform(df[["Age"]]).ravel()

    # Impute fare with Median
    imputer_fare = SimpleImputer(strategy="median")
    imputer_fare.fit(df[["Fare"]])

    # Convert Cabin Type to categorical
    le_cabintype = LabelEncoder()
    df["Cabin_Type"] = le_cabintype.fit_transform(df["Cabin_Type"] )
    print("Mapping:", dict(zip(le_cabintype.classes_, le_cabintype.transform(le_cabintype.classes_))))

    # Convert Survived to bool
    df["Survived"] = df["Survived"].astype(bool)

    return df, imputer_embarked, imputer_age, le_cabintype, imputer_fare

def split_train_val(df):
    
    # for train datasets only - train/validation split
    # train set to fit the model
    # validation set to tune hyperparameters

    y = df["Survived"]
    X = df.drop(columns = ["Survived"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val

def standardise_continuousvar_train(X_train, X_val): 

    # for train datasets only 
    scaler = StandardScaler()
    cont_cols = ["Age", "Fare"]
    # Fit only on training data
    scaler.fit(X_train[cont_cols])

    # Transform train & validation
    X_train.loc[:, cont_cols]  = scaler.transform(X_train[cont_cols])
    X_val.loc[:, cont_cols] = scaler.transform(X_val[cont_cols])

    return X_train, X_val, scaler

def data_cleaning_test(df, imputer_embarked, imputer_age, le_cabintype, imputer_fare, scaler):

    # for test datast only
    df["Embarked"] = imputer_embarked.transform(df[["Embarked"]])[:, 0]
    df["Cabin_Type"] = le_cabintype.transform(df["Cabin_Type"] )
    df["Age"] = imputer_age.transform(df[["Age"]]).ravel()
    df["Fare"] = imputer_fare.transform(df[["Fare"]]).ravel()

    cont_cols = ["Age", "Fare"]
    df[cont_cols] = scaler.transform(df[cont_cols])

    return df 

def get_dataset():

   train_raw = ingest_dataset(type="train")
   train = data_cleaning_general(train_raw)
   train, imputer_embarked, imputer_age, le_cabintype, imputer_fare = data_cleaning_train(train)
   X_train, X_val, y_train, y_val = split_train_val(train)
   X_train, X_val, scaler = standardise_continuousvar_train(X_train, X_val)

   # note that test dataset does not have y values 
   X_test_raw = ingest_dataset(type="test")
   test = data_cleaning_general(X_test_raw)
   X_test = data_cleaning_test(test, imputer_embarked, imputer_age, le_cabintype, imputer_fare, scaler)

   return X_train, X_val, y_train, y_val, X_test, X_test_raw