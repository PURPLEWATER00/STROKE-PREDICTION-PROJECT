import streamlit as st
import pickle 
import pandas as pd


st.set_page_config(
	page_title = 'STROKE PREDICTION MODEL',
	page_icon = '🧠',
    	layout = 'wide',
    	initial_sidebar_state='expanded'
)

model = pickle.load(open('model.pkl','rb'))

st.title = ('STROKE PREDICTION MODEL🧠')

st.sidebar.write('Input features: NOTE: 0 = NO, 1 = YES')
age = st.sidebar.slider('Sepal length', 1, 100, 20)
avg_glucose_level = st.sidebar.slider('Glucose level', 1, 1000, 250)
bmi = st.sidebar.slider('What is your BMI?', 1.0, 100.0, 24.9)
ever_married = st.radio("Are you married?", ('Yes', 'No'))
gender = st.radio("What is your gender?", ('Male', 'Female'))
work_type = st.radio("Which of the following best descibes your work type?", ('Private', 'Self-employed','Govt_job', 'children', 'Never_worked'))
residence_type = st.radio("What is your residence type?", ('Urban', 'Rural'))
smoking_status = st.radio("What is your smoking status?",('formerly smoked', 'never smoked', 'smokes'))

test_df = [[age,avg_glucose_level, bmi, ever_married, gender, work_type, residence_type, smoking_status]]

cat_val = test_df.select_dtypes('object')
test_df = pd.get_dummies(test_df, columns= cat_val.columns)
test_df=test_df.drop(['gender_Female', 'ever_married_No', 'smoking_status_never smoked', 'Residence_type_Rural'], axis=1)

pred_prob = model.predict_proba(x_test)[:,1]

st.subheader('Output')
st.metric('Predicted probability of having a stroke = ', pred_prob, '')



