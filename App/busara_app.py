# import important packages


import streamlit as st
import joblib
import pandas as pd
from os.path import dirname, join, realpath
import joblib
import sklearn

# add banner image
st.header("Busara Mental Health")
st.image("/Users/Jmwaya/Desktop/D-LAB/busara/Busara_Mental_Health.jpeg")
st.subheader(
    """
A simple app that predicts if an individual suffers depression.
"""
)

# form to collect user information 

our_form = st.form(key="busara_form")

ent_employees = our_form.number_input("How many Employees do you have?", min_value=0,max_value=100),  
cons_alcohol = our_form.number_input("How much do you spend on alcohol?", min_value=0,max_value=1000),
fs_adwholed_often = our_form.number_input("How many days without eating?", min_value=0,max_value=30),
fs_chskipm_often = our_form.number_input("How many meals skipped by children?", min_value=0,max_value=3), 
fs_chwholed_often= our_form.number_input("How many days children go without eating ?", min_value=0,max_value=30)
age = our_form.number_input("Enter age", min_value=1, max_value=100)
married = our_form.selectbox("Are you married?",("Yes","No"))
edu = our_form.number_input("How many years of education?", min_value=1,max_value=30)

submit = our_form.form_submit_button(label="make prediction")


# load the model and scaler

with open(
    join(dirname(realpath(__file__)), "/Users/Jmwaya/Desktop/D-LAB/busara/ModelHist.pkl"),
    "rb",
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "/Users/Jmwaya/Desktop/D-LAB/busara/Scaler.pkl"), "rb"
) as f:
    scaler = joblib.load(f)
    
def femaleres_transform(value):
	if value=='Female':
		return 1
	else: 
		return 0
		
def married_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
		
def saved_mpesa_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
    

@st.cache
# function to clean and tranform the input
def preprocessing_data(data, scaler):

    # # Convert the following numerical labels from integer to float
    # float_array = data[["hhsize", "age", "edu","children"]].values.astype(
    #     float
    # )

     # scale our data into range of 0 and 1
    data = scaler.transform(data.values.reshape(-1,1))
    feat_col = ['age', 'ent_employees', 'married', 'edu', 'cons_alcohol','fs_adwholed_often',
       'fs_chskipm_often', 'fs_chwholed_often']

    return pd.DataFrame(data.reshape(-1,8),columns = feat_col)


if submit:


    # collect inputs
    input = {
        "age": age,
        "married": married_transform(married), 
        "ent_employees": ent_employees, 
        "cons_alcohol":cons_alcohol,
        "fs_adwholed_often":fs_adwholed_often,
        "fs_chskipm_often":fs_chskipm_often, 
        "fs_chwholed_often":fs_chwholed_often, 
        "edu":edu
       
    }

    # create a dataframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data, scaler=scaler)

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the RFC task
    st.header("Results")
    if output == 1:
        st.write(
            "You are likely to be depressed with probability of {} 😔".format(
                probability
            )
        )
    elif output == 0:
        st.write(
            "You are likely not to be depressed with probability of {} 😊".format(
                probability
            )
        )


url = "https://twitter.com/@AOgigo"
st.write("Developed with @ by [Group 3](%s)" % url)


