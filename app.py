import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

st.set_page_config(layout='wide')


def build_model(df):
    df = df.loc[:100]
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)
    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    print(predictions_test)
    st.subheader('2. Table of Model Performance')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')
    try:
        with st.markdown('**R-squared**'):
                # Tall
                predictions_test["R-Squared"] = [max(i,0) for i in predictions_test["R-Squared"]]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
                ax1.set(xlim=(0, 1))
    
        st.markdown(imagedownload(plt, 'plot-r2-tall.pdf'), unsafe_allow_html=True)
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0,1))
        plt.xticks(rotation=90)

        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-r2-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE (capped at 50)**'):
            # Tall
                predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)
    except:
        st.write("Please Change the ambigious columns of the dataset")

def filedownload(df, filename):
    # pass
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    # pass
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

st.write("""
# The Machine Learning Algorithm Comparison App
We compare different machine learning algorithms on a dataset
""")
with st.sidebar.header('1. Upload your CSV data'):
    st.image('https://img.freepik.com/free-photo/ai-technology-brain-background-digital-transformation-concept_53876-124672.jpg?q=10&h=200')
    st.title("Auto-Machine-Learning")
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
    st.markdown("Different parameters may not work on different datasets.")

st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100]
        Y = pd.Series(boston.target, name='response').loc[:100]
        df = pd.concat([X, Y], axis=1)
        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))
        build_model(df)
