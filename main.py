#libraries
import streamlit as st
import base64
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json

# interact with FastAPI endpoint
backend = "https://api-ultrasound-image-classificator.onrender.com/classification"


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    return r


#open images
img_logo = Image.open('img/Ultrasound_Task2.png')

#starting application
st.set_page_config(page_title="Ultrasound Plane Classificator", page_icon="🩺", layout="wide")
a, b, c=st.columns([0.1, 1, 0.1])
a.write("")
b.image(img_logo,  width=1900)
c.write("")

selected = option_menu(None, ["Documentation", 'Predictor','Settings'], icons=['house', 'cloud-upload', 'gear'], menu_icon="cast", default_index=0, orientation="horizontal")

hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
st.markdown(hide_img_fs, unsafe_allow_html=True)

if selected=="Documentation":
    st.title("AI Tool to predict classes of ultrasound neonatal plane")
    st.subheader("Scopri a che cosa serve e come utilizzare il tool")

elif selected == "Predictor":

    st.title("MobileNet Image classification model")
    st.subheader("The image classification model classifies image in 3 categories: Brain, Cervix, Throrax")

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
      
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    c0,  c1, c2,  c3,  c4= st.columns([0.1, 1, 0.1, 1, 0.1])
    if upload is not None:
      im= Image.open(upload)
      """img= np.asarray(im)
      image= cv2.resize(img,(224, 224))
      img= preprocess_input(image)
      img= np.expand_dims(img, 0)"""
      c0.write("")
      c1.header('Input Image')
      c1.image(im)
      c3.write("")
     
    
    c1.write("")
    button=c1.button("Predict class")
    if button==True:
        #predict_result=prediction_ml(img)
        segments = process(upload, backend)
        #mb_pred_round = np.round(predict_result)
        #mb_pred_classes = np.argmax(predict_result, axis=1)
        c3.header('Output')
        c3.subheader('Predicted class :')
        classes=np.array(['Brain','Thorax','Cervix'])
        c3.write(segments.text)
        resp_js=segments.json()
        data = json.loads(resp_js)
        perc_brain=round(float(data['BRAIN']), 4)*100
        perc_thx=round(float(data['THORAX']), 4)*100
        perc_cerx=round(float(data['CERVIX']),  4)*100
        
        if perc_brain>perc_thx and perc_brain>perc_cerx:
            c3.subheader("La classe di appartenza è BRAIN con una probabilità del: "+str(perc_brain)+"%")
        elif perc_thx>perc_brain and perc_thx>perc_cerx:
            c3.subheader("La classe di appartenza è THORAX con una probabilità del: "+(str(perc_thx))+"%")
        elif perc_cerx>perc_brain and perc_cerx>perc_thx:
            c3.subheader("La classe di appartenza è CERVIX con una probabilità del: "+str(perc_cerx)+"%")
        else:
            c3.write("error")
            
        #chart_data = pd.DataFrame(predict_result, columns=['Brain','Thorax','Cervix'])
        #c3.write(chart_data)
      
        #c4.bar_chart(data=mb_pred_round, use_container_width=True)
