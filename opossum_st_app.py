"""
Created on Monday 05 Nov. 2022
@author: Charley ∆. Lebarbier
"""

# Import librairies
import joblib
import numpy as np
import streamlit as st


#--------------------------------#
#load model ML - Social Network
model = joblib.load('model/opossum_joblib')


@st.cache    # Caching the model for fast loading
def predict(hdlngth, skullw, totlngth, eye, chest, belly):
  """
    Use the ML Model to predict Future Purchaser
    Get 6 params : hdlngth, skullw, totlngth, eye, chest, belly
  """
  input_predict = np.array([hdlngth, skullw, totlngth, eye, chest, belly])
  input_predict = input_predict.reshape(1, -1)

  prediction = (model['Model']).predict((model['Scaler']).transform(input_predict))
  prediction = str(prediction)[2:-2]  # del the double bracket

  return int(round(float(prediction), ndigits=0))



##############################################################
########################## STREAMLIT #########################
##############################################################

# METADATA WEBAPP
st.set_page_config(
                    page_title = "Opossum Age Prediction",
                    page_icon = ":crystal_ball:",
                    layout = "wide")

### Background
page_bg_img = f"""
  <style>
    .stApp {{
    background-image: url("https://github.com/CharleyDL/Opossum/blob/main/img/bckgrd.jpg?raw=true");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
  </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# HEADER AND WEB CONTENT
st.markdown("<h1 style='text-align: center;'>Possum Age Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

### First Row 
with col1:
  st.header("Head Length")
  hdlngth = st.number_input("", min_value=70, max_value=110)

with col2:
  st.header("Skull Width")
  skullw = st.number_input("", min_value=50, max_value=70)

with col3:
  st.header("Total Length")
  totlngth = st.number_input("", min_value=70, max_value=100)

### Second Row
with col1:
  st.header("Eye Size")
  eye = st.number_input("", min_value=10, max_value=20)

with col2:
  st.header("Chest Size")
  chest = st.number_input("", min_value=20, max_value=35)

with col3:
  st.header("Belly Size")
  belly = st.number_input("", min_value=25, max_value=40)


#Button
st.markdown("----", unsafe_allow_html=True)
columns = st.columns((2, 1, 2))


if columns[1].button('Possum Age'):
  prediction = predict(hdlngth, skullw, totlngth, eye, chest, belly)
  st.info(f"---- It's {prediction} year(s) old", icon='🎂')