import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Cardiovascular Disease Prediction')
st.markdown('This mini app analyse data from Framingham Cardiovascular Disease Dataset '
            'and predicts who are likely to get this cardiac disease.')

from PIL import Image

img = Image.open('heart.jpg')
st.image(img, use_column_width=True)

st.header('Framingham Cardiovascular Disease Dataset')
st.markdown('The data has been cleaned and the raw data can be obtained from '
            '[here]( https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data/ ).')

chd_raw = pd.read_csv('coronary.csv')


def user_input():
    st.sidebar.header('User Input')

    age_factor = st.sidebar.slider('Age', 32, 71, 50)
    chol_factor = st.sidebar.slider('Total Cholesterol', 107, 600, 350)
    bp_factor = st.sidebar.slider('Blood Pressure', 83.0, 296.0, 120.0)
    diabp_factor = st.sidebar.slider('Diastolic Blood Pressure', 48.0, 143.0, 80.0)
    bmi_factor = st.sidebar.slider('BMI', 15.0, 57.0, 22.0)
    glu_factor = st.sidebar.slider('Glucose Level', 40, 394, 170)

    data = {'Age': age_factor,
            'TotalChol': chol_factor,
            'BloodPressure': bp_factor,
            'diaBP': diabp_factor,
            'BMI': bmi_factor,
            'Glucose': glu_factor}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input()

# Combines user's input with the entire datasets
chd = chd_raw.drop(columns=['TenYearCHD'])
st.write(chd.head())
df1 = pd.concat([input_df, chd], axis=0)
df1 = df1[:1]

# Display User's choice
st.subheader('Your input')
st.write(df1)

df2 = chd_raw.copy()

# Separating X and y
X = df2.drop('TenYearCHD', axis=1)
y = df2['TenYearCHD']

# Building random forest model
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df1)
prediction_proba = clf.predict_proba(df1)

st.subheader('In the next ten years, the person with the above feature will likely...')
chd_pred = np.array([1, 0])
yes = chd_pred[prediction]
if yes == 0:
    st.write('have Coronary Disease')
else:
    st.write('not have Coronary Disease')

st.subheader('Prediction Probability')
st.write(prediction_proba)
