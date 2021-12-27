import streamlit as st
import pickle
import pandas as pd
import warnings
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

warnings.filterwarnings('ignore')

# Load model

@st.cache
def load_data(csv_filepath="data/Test.csv"):

    df = pd.read_csv(csv_filepath, low_memory=False,
                          parse_dates=["saledate"])

    return df


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


def project_info_text():
    st.write("""
        # Predicting the Sale Price of Bulldozers using ML

        In this notebook, we're going to predict the sales price of bulldozers.
        
        ## 1. Problem definition
        
        > How well can we predict the future sale price of a bulldozer, given its attributes and historical sales of bulldozers at auctions
        
        ## 2. Data
        
        The data is downloaded from the Kaggle Bluebook for Bulldozers competition:
        
        There are 3 main datasets:
        
        * Train.csv is the training set, which contains data through the end of 2011.
        * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012.
        * Test.csv is the test set, which contains data from May 1, 2012 - November 2012.
        
        ## 3. Evaluation
        
        The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
        
        For more information on the evaluation of this project check:
        https://www.kaggle.com/c/bluebook-for-bulldozers/data
        
        
        ## 4. Features
        
        Kaggle provides a data dictionary detailing all of the features of the dataset. You can view it here:
        https://docs.google.com/spreadsheets/d/1_fSxZuMwTByx5oD6se1Ji1EUezIdCtfmJi6e7ltDw-0/edit#gid=590674478

        """)

def result_df_download(input_df, result):
    result_df = pd.DataFrame()
    result_df["SalesID"] = input_df["SalesID"]
    result_df["SalesPrice"] = result
    st.dataframe(pd.DataFrame(result_df))

    st.download_button(
        "Download",
        convert_df(result_df),
        "result.csv",
        "text/csv")


def main():
    choice = st.sidebar.radio("Select action:",
                                  options=["Project Info", "Predict with ML model"])

    if choice == "Project Info":
        project_info_text()

    if choice == "Predict with ML model":
        small_model = pickle.load(open("./model/small_model.pkl", "rb"))

        df = load_data()

        # Predict with test data
        st.write("### Predicting with sample data")
        st.dataframe(df.head(1))
        if st.button("Predict with sample"):
           result = small_model.predict(preprocess_data(df).head(1))
           st.success(f"Predicted Sales Price: ${result[0]:.2f}")
        st.write("---")

        # Predict with user data
        st.write("### Predicting with your own data")

        def user_file_operation():

            file_path = "./data/Test.csv"

            with open(file_path, 'rb') as my_file:
                st.download_button(label='Download sample CSV', data=my_file, file_name='sample.csv',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            user_file = st.file_uploader("Upload a file containing your features")

            if user_file:
                user_df = load_data(user_file)
                if st.button("Predict"):
                    result = small_model.predict(preprocess_data(user_df))
                    result_df_download(user_df, result)



        user_file_operation()

if __name__ == '__main__':
    main()
