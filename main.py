# import streamlit as st
# import pandas as pd
# import numpy as np
# from os import path
# # st.title("hello")
# #
# # st.write("hello world")
# # creating a dataframe
# # df_Data = pd.DataFrame({'Column1' : [1,2,3,5],
# #                         'Column2': ['a','b','c','d']})
# #
# # st.write(df_Data) #displaying the dataframe we creates
#
# st.title('Iris dataset')
# df_iris = pd.read_csv(path.join("Data","iris.csv"))
# #Root/Data/iris.csv = file path structure
# st.write(df_iris)
# st.scatter_chart(df_iris[['sepal_length','sepal_width']])
#
# #df_map = pd.DataFrame(np.array([[15.131588036403969, 38.21987373955566]]),
#                       #columns = ["LAT","LON"])
# #st.map(df_map)
#
# petal_length = st.slider("Choose a petal length", min_value =1.0, max_value=6.9)
# petal_width = st.slider("Choose a petal width")
# sepal_length = st.slider("Choose a sepal length")
# sepal_width = st.slider("Choose a sepal width")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page title
st.title("🌸 Iris Species Predictor")
st.write("A simple machine learning app to predict Iris flower species")


# Load data
@st.cache_data
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "Data", "iris.csv")
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load model
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "Model", "iris.pkl")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load data and model
data = load_data()
model = load_model()

if data is not None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["View Data", "Make Prediction"])

    if page == "View Data":
        st.header("📊 Dataset Overview")

        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            st.metric("Species", data.iloc[:, -1].nunique())

        # Show data
        st.subheader("Data Sample")
        st.dataframe(data.head(10))

        # Show species count
        st.subheader("Species Distribution")
        species_count = data.iloc[:, -1].value_counts()
        st.bar_chart(species_count)

    elif page == "Make Prediction":
        st.header("🔮 Predict Iris Species")

        if model is not None:
            st.write("Enter the measurements to predict the species:")

            # Input sliders
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.0, 0.1)

            # Make prediction
            if st.button("Predict Species"):
                # Prepare input
                input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

                # Get prediction
                prediction = model.predict(input_data)[0]

                # Show result
                st.success(f"Predicted Species: **{prediction}**")

                # Show probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_data)[0]
                    prob_df = pd.DataFrame({
                        'Species': model.classes_,
                        'Probability': probabilities
                    })
                    st.write("Prediction Probabilities:")
                    st.bar_chart(prob_df.set_index('Species'))

else:
    st.info("Please upload the iris.csv file in the Data folder to get started.")
