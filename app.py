import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

st.write("""
# Predicting the Sale Price of Bulldozers using ML
In this notebook, we're going to predict the sales price of bulldozers.

## Features
Kaggle provides a data dictionary detailing all of the features of the dataset. You can view it here: https://docs.google.com/spreadsheets/d/1_fSxZuMwTByx5oD6se1Ji1EUezIdCtfmJi6e7ltDw-0/edit#gid=590674478
""")

st.write("---")


small_model = pickle.load(open("./model/small_model.pkl", "rb"))




@st.cache()
def load_test_data():
    df_test = pd.read_csv("data/Test.csv", low_memory=False,
                          parse_dates=["saledate"])

    return df_test

def preprocess_data(df):
    '''
    Performs transformation
    '''

    df["sale_year"] = df.saledate.dt.year
    df["sale_month"] = df.saledate.dt.month
    df["sale_day"] = df.saledate.dt.day
    df["sale_dayOfWeek"] = df.saledate.dt.dayofweek
    df["sale_dayOfYear"] = df.saledate.dt.dayofyear

    df.drop("saledate", axis=1, inplace=True)

    # Find columns which contain strings

    # Check for which numeric columns have null values

    for label, content in df.items():

        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()

        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label + "_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())

        if not pd.api.types.is_numeric_dtype(content):
            df[label + "_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1

    df["auctioneerID_is_missing"] = False

    return df


def main():
    df = load_test_data()
    st.dataframe(df.head(1))
    if st.button("Predict with test data"):
       df_test = preprocess_data(df)
       result = small_model.predict(df_test.head(1))
       st.success(f"Predicted Sales Price: ${result[0]:.2f}")

if __name__ == '__main__':
    main()
