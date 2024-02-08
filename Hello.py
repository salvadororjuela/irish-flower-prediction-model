import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
         
This app predicts the **Iris Flower** type!
""")

st.sidebar.header("User Input Parameters")

"""
Function that accepts the input parameters using sliders to select the values. 
Then creates a dictionary ("data") with the input data and then creates 
a Pandas data frame which is returned to predict the Irish Flower type
"""
def user_input_features():
    # The fists two digits are the max and min range, and the third 
    # corresponds to the default value. It is also displayed onscreen
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petalwidth", 0.1, 2.5, 0.2)
    data = {"sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length, 
            "petal_width": petal_width        
    }
    features = pd.DataFrame(data, index = [0])
    return features

# Store the input values when calling the function user_input_features()
df = user_input_features()

st.subheader("User Input Parameters")
# Displays the data frame with the input values
st.write(df)

# Loads the iris dataset from scikit learn
iris = datasets.load_iris()
# X corresponds to the four features or columns of the data set
# (sepal_lenght, sepal_width, petal_lenght, petal_width)
X = iris.data
# Y corresponds to the class index number (0, 1, 2)
Y = iris.target

# Clasiffier on which we apply the prediction
clf = RandomForestClassifier()
# Model training
clf.fit(X, Y) 
# Prediction
prediction = clf.predict(df)
# Prediction probability
prediction_proba = clf.predict_proba(df)

st.subheader("Class labels and their corresponding index number")
# Display all the class names
st.write(iris.target_names)

st.subheader("Prediction")
# Displays the predicted class name
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
# Displays the probability of been into each of the classes 
st.write(prediction_proba)
