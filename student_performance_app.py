import streamlit as st
import pandas as pd
import pickle

# Load your trained RandomForest model
with open('rf_classifier_model.pkl', 'rb') as file:
    clf_model = pickle.load(file)

st.title("Student Performance Prediction and Personalized Learning Plan")

# Function to map yes/no to 1/0
def yes_no_to_int(x):
    return 1 if x == 'yes' else 0

# Collect user input
def user_input_features():
    school = st.selectbox('School', ['GP', 'MS'])
    sex = st.selectbox('Sex', ['M', 'F'])
    age = st.number_input('Age', min_value=15, max_value=22, value=17)
    address = st.selectbox('Address', ['U', 'R'])
    famsize = st.selectbox('Family Size', ['LE3', 'GT3'])
    Pstatus = st.selectbox('Parent Status', ['T', 'A'])
    Medu = st.slider('Mother Education (0-4)', 0, 4, 2)
    Fedu = st.slider('Father Education (0-4)', 0, 4, 2)
    reason = st.selectbox('Reason to choose this school', ['home', 'reputation', 'course', 'other'])
    guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
    traveltime = st.slider('Travel time to school (1-4)', 1, 4, 1)
    studytime = st.slider('Weekly study time (1-4)', 1, 4, 2)
    failures = st.slider('Past class failures', 0, 3, 0)
    schoolsup = st.selectbox('School support', ['yes', 'no'])
    famsup = st.selectbox('Family support', ['yes', 'no'])
    paid = st.selectbox('Extra paid classes', ['yes', 'no'])
    activities = st.selectbox('Extra-curricular activities', ['yes', 'no'])
    nursery = st.selectbox('Attended nursery school', ['yes', 'no'])
    higher = st.selectbox('Wants higher education', ['yes', 'no'])
    internet = st.selectbox('Internet access at home', ['yes', 'no'])
    romantic = st.selectbox('In a romantic relationship', ['yes', 'no'])
    famrel = st.slider('Family relationship quality (1-5)', 1, 5, 3)
    freetime = st.slider('Free time after school (1-5)', 1, 5, 3)
    goout = st.slider('Going out with friends (1-5)', 1, 5, 3)
    Dalc = st.slider('Workday alcohol consumption (1-5)', 1, 5, 1)
    Walc = st.slider('Weekend alcohol consumption (1-5)', 1, 5, 1)
    health = st.slider('Current health status (1-5)', 1, 5, 3)
    absences = st.number_input('Number of school absences', min_value=0, max_value=93, value=0)
    G1 = st.slider('Grade in first period (0-20)', 0, 20, 10)
    G2 = st.slider('Grade in second period (0-20)', 0, 20, 10)
    G3 = st.slider('Final grade (0-20)', 0, 20, 10)

    data = {
        'school': school,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': nursery,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences,
        'G1': G1,
        'G2': G2,
        'G3': G3
    }
    features = pd.DataFrame(data, index=[0])

    # Map yes/no columns to 1/0
    yes_no_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in yes_no_cols:
        features[col] = features[col].apply(yes_no_to_int)

    # If your model expects categorical columns as numbers, you might need to encode 'school', 'sex', 'address', etc.
    # Example label encoding:
    features['school'] = features['school'].map({'GP': 0, 'MS': 1})
    features['sex'] = features['sex'].map({'F': 0, 'M': 1})
    features['address'] = features['address'].map({'U': 0, 'R': 1})
    features['famsize'] = features['famsize'].map({'LE3': 0, 'GT3': 1})
    features['Pstatus'] = features['Pstatus'].map({'T': 0, 'A': 1})
    features['reason'] = features['reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
    features['guardian'] = features['guardian'].map({'mother': 0, 'father': 1, 'other': 2})

    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

if st.button('Predict'):
    try:
        prediction = clf_model.predict(input_df)[0]
        st.success(f'Prediction: {"Pass" if prediction == 1 else "Fail"}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
