# NCVPRIPG-2024 Auto Eval Challenge

import cv2

import numpy as np #Image Processing 
import pandas as pd

import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing


#title
st.title("NCVPRIPG'24: Auto-Eval Challenge")

#subtitle
st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit`")

st.markdown("")

#image uploader
image = st.file_uploader(label = "Please upload your image here",type=['png','jpg','jpeg'])


@st.cache
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 

reader = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):
    #    result = reader.readtext(np.array(input_image))
    #    result_text = [] #empty list for results
    #    for text in result:
    #         result_text.append(text[1])
    
        res = reader.readtext(np.array(input_image))

        for t in res:
            bbox, text, score = t
            l_bbox = bbox[0][0]
            l_bbox1 = bbox[0][1]
            r_bbox = bbox[2][0]
            r_bbox1 = bbox[2][1]
            img_out = cv2.rectangle(np.array(input_image),
                                    (int(l_bbox), int(l_bbox1)),
                                    (int(r_bbox), int(r_bbox1)), 
                                    (255,0,0),5)
            
        st.image(img_out) #display image
        
        all_text = []
        for i in range(len(res)):
             text = res[i][1]
             all_text.append(text)
        # st.write(result_text)
        st.write(all_text)
    #st.success("Here you go!")

        all_text = []
        lb = []
        lb1 = []
        rb = []
        rb1 = []
        prob = []
        
        for t in res:
            bbox, text, score = t
            l_bbox = bbox[0][0]
            l_bbox1 = bbox[0][1]
            r_bbox = bbox[2][0]
            r_bbox1 = bbox[2][1]
            img_out = cv2.rectangle(img, (int(l_bbox), int(l_bbox1)), (int(r_bbox), int(r_bbox1)), (255,0,0),5)
            all_text.append(text)
            lb.append(l_bbox)
            lb1.append(l_bbox1)
            rb.append(r_bbox)
            rb1.append(r_bbox1)
            prob.append(score)

    df = pd.DataFrame({'ocr_word' : all_text,
                   'lb' : lb, 
                   'lb1':lb1,
                   'rb':rb,
                   'rb1':rb1,
                   'prob':prob})
    
    trsld = np.array(input_image).shape[0] * 0.8
    df_sub = df[(df['lb'] >= trsld) | (df['rb'] >= trsld)].reset_index(drop = True)
    
    st.write(df_sub)

    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made with ❤️ by Team: IIT Jodhpur")





