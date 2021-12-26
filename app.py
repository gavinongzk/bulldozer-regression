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



# Load model


@st.cache
def load_data(csv_filepath="data/Test.csv"):

    df_test = pd.read_csv(csv_filepath, low_memory=False,
                          parse_dates=["saledate"])

    return df_test


@st.cache
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

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


def main():

    small_model = pickle.load(open("./model/small_model.pkl", "rb"))
    df = load_data()

    # Predict with test data
    st.write("### Predicting with test data")
    st.dataframe(df.head(1))
    if st.button("Predict test"):
       df_test = preprocess_data(df)
       result = small_model.predict(df_test.head(1))
       st.success(f"Predicted Sales Price: ${result[0]:.2f}")
    st.write("---")
    st.write("### Predicting with your own data")
    file_path = "./data/Test.csv"

    # Predict with user data
    with open(file_path, 'rb') as my_file:
        st.download_button(label = 'Download sample CSV', data = my_file, file_name = 'sample.csv', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    user_file = st.file_uploader("Upload a file containing your features")
    if user_file:
        user_df = load_data(user_file)
        if st.button("Predict"):
            proc_user_df = preprocess_data(user_df)
            result = small_model.predict(proc_user_df)
            result_df = pd.DataFrame()
            result_df["SalesID"] = user_df["SalesID"]
            result_df["SalesPrice"] = result
            st.dataframe(pd.DataFrame(result_df))

if __name__ == '__main__':
    main()
