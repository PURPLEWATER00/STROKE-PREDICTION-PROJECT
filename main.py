import streamlit as st
import pickle 
import pandas as pd


st.set_page_config(
	page_title = 'STROKE PREDICTION MODEL',
	page_icon = '🧠',
    	layout = 'wide',
    	initial_sidebar_state='expanded'
)


def set_video_background():
    video_url = 'https://www.youtube.com/watch?v=fG_X-cx5Szg&pp=ygUXdmlkZW8gb2Ygb2NlYW4gaGlnaCByZXM%3D'

    video_html = f'''
    <video autoplay loop muted style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1;">
        <source src="{video_url}" type="video/mp4">
    </video>
    '''
    st.markdown(video_html, unsafe_allow_html=True)

def main():
    set_video_background()

    # Your Streamlit app content goes here
    st.write("Streamlit App with Video Background")
    st.write("Welcome to my app!")


model = pickle.load(open('model.pkl','rb'))

st.header('🧠  STROKE PREDICTION MODEL')

st.sidebar.write('Input features:')
age = st.sidebar.slider('Age:', 1, 100, 20)
avg_glucose_level = st.sidebar.slider('Glucose level', 1.0, 1000.0, 250.0)
bmi = st.sidebar.slider('What is your BMI?', 1.0, 100.0, 24.9)
ever_married = st.radio("Are you married?", ('Yes', 'No'))
gender = st.radio("What is your gender?", ('Male', 'Female'))
work_type = st.radio("Which of the following best descibes your work type?", ('Private', 'Self-employed','Govt_job', 'children', 'Never_worked'))
residence_type = st.radio("What is your residence type?", ('Urban', 'Rural'))
smoking_status = st.radio("What is your smoking status?",('formerly smoked', 'never smoked', 'smokes'))


ever_married_indx = 1 if ever_married == "Yes" else 0
gender_indx = 1 if gender == "Male" else 0

work_type_input = work_type
work_type_indx  = {
    "Private": 0,
    "Self-employed": 0,
    "Govt_job": 0,
    "children": 0,
    "Never_worked": 0,
}
work_type_indx[work_type_input] = 1

residence_type_indx  = 1 if residence_type == "Urban" else 0

smoking_status_input = smoking_status
smoking_status_formerly_smoked_indx = 1 if smoking_status_input == "formerly smoked" else 0
smoking_status_smokes_indx = 1 if smoking_status_input == "smokes" else 0

data = {
    "age": [age],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender_Male": [gender_indx ],
    "ever_married_Yes": [ever_married_indx],
    "work_type_Govt_job": [work_type_indx ["Govt_job"]],
    "work_type_Never_worked": [work_type_indx ["Never_worked"]],
    "work_type_Private": [work_type_indx ["Private"]],
    "work_type_Self-employed": [work_type_indx ["Self-employed"]],
    "work_type_children": [work_type_indx ["children"]],
    "Residence_type_Urban": [residence_type_indx ],
    "smoking_status_formerly smoked": [smoking_status_formerly_smoked_indx ],
    "smoking_status_smokes": [smoking_status_smokes_indx ]
}

test_df = pd.DataFrame(data)


pred_prob = model.predict_proba(test_df)[:,1]

st.subheader('Output')
st.metric('Predicted probability of having a stroke = ', pred_prob, '')

if __name__ == '__main__':
    main()



