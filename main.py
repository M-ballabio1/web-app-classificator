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

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# interact with FastAPI endpoint
#backend = "https://api-ultrasound-image-classificator.onrender.com/classification"
backend = "https://api-ultrasound-classificator-cloud-run-pa6vji5wfa-ew.a.run.app/classification"


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    return r


#open images
img_logo = Image.open('img/Ultrasound_Task2.png')
img_icon = Image.open('img/icon_ultra.png')
img_pipe = Image.open('img/mlops_pipe_fin.png')

#starting application
st.set_page_config(page_title="Ultrasound Plane Classificator", page_icon="🩺", layout="wide")

# Aggiungi questa linea per inizializzare la variabile di sessione
if 'call_count' not in st.session_state:
    st.session_state.call_count = 0

st.image(img_logo)

#sidebar
with st.sidebar:
    a,b,c=st.columns([0.45,1,0.1])
    b.image(img_icon, width=100)
    st.header("")
    selected = option_menu("Menu Choice", ["Documentation", 'Predictor','Report Bug'], icons=['house', 'cloud-upload', 'gear'], menu_icon="cast", default_index=0, orientation="vertical")

    st.write("")

    st.markdown("""<hr style="height:5px;border:none;color:#e4e3e3;background-color:#e4e3e3;" /> """, unsafe_allow_html=True)
    
    st.sidebar.header("Info")
    st.info(
    """
    Questa è una webapp che consente di interagire con l'api ultrasound classificator
    
    API URL: https://api-ultrasound-classificator-cloud-run-pa6vji5wfa-ew.a.run.app/
    """
    )

    st.sidebar.header("Support")
    st.sidebar.info(
        """
        Per eventuali problemi nell'utilizzo app rivolgersi a: matteoballabio99@gmail.com
        """
    )

hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
st.markdown(hide_img_fs, unsafe_allow_html=True)

# documentation
if selected=="Documentation":
    st.title("AI-Driven Ultrasound Classification Assistant")

    with st.expander("**Operationalizing Machine Learning Models for Standard Fetal Ultrasound Plane Classification: A Framework for Applying MLOps in Clinical Practice ⬇️**"):
        a,b,c=st.columns([1,1.5,1])
        b.subheader("Abstract")
        b.write("""Integrating Machine Learning (ML) models into healthcare can revolutionize
                        medical diagnosis and treatment. However, implementing and maintaining
                        ML models in clinical practice can be challenging. This paper explores the ap-
                        plication of Machine Learning Operations (MLOps) to a specific medical use
                        case, showing its potential benefits in this context. We focus on classifying
                        standard fetal planes from gynecological ultrasound images, a widely explored
                        and crucial task for assessing and monitoring fetal growth. Introducing our
                        Health-MLOps (H-MLOps) pipeline, which includes data acquisition, model
                        training, monitoring, explainability, and sustainability, we provide practical
                        guidance to researchers and clinicians interested in incorporating MLOps in
                        the clinical practice to improve the reliability and efficiency of ML models in
                        healthcare.""")
        b.write("Keywords: MLOps, Machine learning, Software engineering, Fetal ultrasound, Medical image analysis")
    
    st.subheader("Explanation of tool and MLOps Pipeline")
    st.info("""
            **Welcome to the Medical Doctor Client Web Application!**

            _**Goal:**_ This web application simulates the interaction of a medical doctor client with a Web service API hosted on Google Cloud Run.
            The API's primary objective is to process the medical images received from the client using a powerful Deep Learning algorithm. It then provides a JSON response containing the probabilities of classification for the image across three classes.

            _**User Action:**_ As a medical doctor, you can upload an image through the interface, and the API will analyze it to determine the likelihood of belonging to each of the specified classes.
            
            Please note that this is a simulated environment, and the results presented here are for demonstrative purposes only.
            Enjoy exploring the capabilities of our Deep Learning-based medical image classification system!
            """, icon="ℹ️")
    with st.container():
        st.image(img_pipe)
        st.caption(""":blue[Figure 4: 1) Data ingestion is done automatically and continuously through Google Drive.
                            2) Neptune.ai is used for comprehensive and recurring management of experiments and
                            monitoring (2A) of data, models, metrics, and continuous integration. 3) The most promis-
                            ing models are then sent to a Git repository. This process can be streamlined using GitHub
                            Actions, which acts as a process orchestrator to synchronize repository update operations
                            and automate pipeline creation, testing, and deployment. 4) Finally, the model is deliv-
                            ered as a Docker container and then hosted by the Cloud Run platform. To monitor the API
                            (4A), we can use the Cloud Run platform, which allows us to monitor API usage, call limit, and user interaction with the IP]""")
    
    st.header("")

    st.subheader("Test the tool here! ")
    with open("img/Brain_Plane.png", "rb") as file:
        st.download_button(label="Download image ⬇️",
                data=file,
                file_name="Brain_Plane.png",
                mime="image/png"
            )
    


    #st.divider()

elif selected == "Predictor":

    # Controlla il numero massimo di chiamate prima di procedere
    if st.session_state.call_count >= 3:
        st.warning("Hai raggiunto il numero massimo consentito di chiamate all'API.")
        st.stop()

    st.title("MobileNet Image classification model 🚀")
    st.subheader("The image classification model classifies image in 3 categories: Brain, Cervix, Throrax")

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
      
    uploaded_image_flag = False

    upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
    c0, c1, c2, c3, c4 = st.columns([0.1, 1, 0.1, 1, 0.1])

    if upload is not None:
        uploaded_image_flag = True
        im = Image.open(upload)
        c0.write("")
        c1.header('Input Image')
        c1.image(im)
        c3.write("")

    c1.write("")

    if uploaded_image_flag and c1.button("Predict class ➡️"):
        st.session_state.call_count += 1
        with st.spinner("Predicting..."):
            segments = process(upload, backend)
        #mb_pred_round = np.round(predict_result)
        #mb_pred_classes = np.argmax(predict_result, axis=1)
        c3.header('Output')
        c3.subheader('Predicted class:')
        classes = np.array(['Brain', 'Thorax', 'Cervix'])
        c3.write(segments.text)
        resp_js = segments.json()
        data = json.loads(resp_js)
        perc_brain = round(float(data['BRAIN']), 4) * 100
        perc_thx = round(float(data['THORAX']), 4) * 100
        perc_cerx = round(float(data['CERVIX']), 4) * 100

        if perc_brain > perc_thx and perc_brain > perc_cerx:
            c3.subheader("Class: BRAIN, Probability: " + str(perc_brain) + "%")
        elif perc_thx > perc_brain and perc_thx > perc_cerx:
            c3.subheader("Class: THORAX, Probability: " + str(perc_thx) + "%")
        elif perc_cerx > perc_brain and perc_cerx > perc_thx:
            c3.subheader("Class: CERVIX, Probability: " + str(perc_cerx) + "%")
        else:
            c3.write("Error")

elif selected=="Report Bug":
    st.title("Bug reporting")
    form = st.form(key="annotation", clear_on_submit=True)
    with form:
        cols = st.columns((1, 1))
        author = cols[0].text_input("Report author:")
        bug_type = cols[1].selectbox(
            "Bug type:", ["Front-end", "Back-end", "Data related", "404"], index=2
        )
        comment = st.text_area("Comment:")
        cols = st.columns(2)
        date = cols[0].date_input("Bug date occurrence:")
        bug_severity = cols[1].slider("Bug severity:", 1, 5, 2)
        submitted = st.form_submit_button(label="Submit")

        if submitted:

            def send_email(sender_email, sender_password, receiver_email, subject, body):
                # SMTP configuration for Gmail
                smtp_server = 'smtp.gmail.com'
                smtp_port = 587

                # Create a secure connection to the SMTP server
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()

                try:
                    # Log in to your Gmail account
                    server.login(sender_email, sender_password)

                    # Create the email message
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = receiver_email
                    msg['Subject'] = subject

                    # Attach the body of the email
                    msg.attach(MIMEText(body, 'plain'))

                    # Send the email
                    server.sendmail(sender_email, receiver_email, msg.as_string())

                except Exception as e:
                    print(f"Failed to send email. Error: {e}")
                finally:
                    # Close the connection to the SMTP server
                    server.quit()

            # Replace the following variables with your own values
            sender_email = 'matteoballabio99@gmail.com'
            # Load the secret from secrets.toml
            sender_password = st.secrets["my_email"]["password"]                      #A2F in security --> in fondo per passw generated
            sender_password = 'otityzbpoehysqlf'
            receiver_email = 'archiviazione.dati1999@gmail.com'
            subject = 'Reporting Error Web app ultrasound classificator'
            body = f"""Ciao Matteo,

sono {author}, in data {date} sto riportando un bug di tipo {bug_type} e severità {str(bug_severity)}.

Ecco il mio commento in merito:
{comment}

Ciao."""

            send_email(sender_email, sender_password, receiver_email, subject, body)
            st.success("Bug sent")


