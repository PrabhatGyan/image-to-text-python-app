# Auto Eval NCVPRIPG-24

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import os
import time
import cv2
import json
import Levenshtein

import pandas as pd
import numpy as np
from pathlib import Path

from tqdm import tqdm

import matplotlib.pyplot as plt

# Multiple print statments in a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Adjust the Display options for number of rows and columns
pd.set_option("display.max_columns", 500)
pd.set_option("display.min_rows", 500)

# Supress the warnings
import warnings
warnings.filterwarnings("ignore")

# Let's pick the desired backend
os.environ['USE_TORCH'] = '1'


import streamlit as st  #Web App
from PIL import Image #Image Processing

#title
st.title("NCVPRIPG'24: Auto-Eval Challenge")

#subtitle
st.markdown("## Optical Character Recognition - Using `DocTR`, `streamlit`")

st.markdown("")

#image uploader
image = st.file_uploader(label = "Please upload your image here.",
                         type=['png','jpg','jpeg'])

def json_out(path_json_output):
    data = json.load(open(path_json_output))

    pg_info = data['pages'][0]

    words_info = []
    for block_info in pg_info['blocks']:
        for line_info in block_info['lines']:
            for word_info in line_info['words']:
                words_info.append(word_info)
    words_info.sort( key=lambda x: x['geometry'][0][0])
    first_col = words_info[:11]
    first_col.sort( key=lambda x: x['geometry'][0][1])
    words_info.sort( key=lambda x: x['geometry'][1][0])
    last_col = words_info[-11:]
    last_col.sort( key=lambda x: x['geometry'][0][1])

    true_false_with_sn = {}
    for w in first_col:
        try:
            true_false_with_sn[int(w['value'])] = []
        except:
            pass
    for w in last_col:
        closest = None
        dist = 1
        cent = (w['geometry'][0][1]+w['geometry'][1][1])/2
        for c in first_col:
            if abs((c['geometry'][0][1]+c['geometry'][1][1])/2 - cent) < dist:
                dist = abs((c['geometry'][0][1]+c['geometry'][1][1])/2 - cent)
                closest = c
        try:
            true_false_with_sn[int(closest['value'])].append(w['value'])
        except:
            pass

    return true_false_with_sn

def tf(row):
    if row['min_dist'] == row['edit_dist_true']:
        T_F_Out = 'True'
    elif row['min_dist'] == row['edit_dist_false']:
        T_F_Out = 'False'

    return T_F_Out

def df_output(post_processed_output):
    df_out = pd.DataFrame(post_processed_output.items(), columns = ['SN','T_F'])
    df_out['T_F'] = df_out['T_F'].str[0]
    df_out['T_F_lower'] = df_out['T_F'].str.lower()
    df_out["edit_dist_true"] = df_out.apply(lambda row: Levenshtein.ratio(str(row["T_F_lower"]), str('true')), axis=1)
    df_out["edit_dist_false"] = df_out.apply(lambda row: Levenshtein.ratio(str(row["T_F_lower"]), str('false')), axis=1)
    df_out['min_dist'] = df_out[['edit_dist_true','edit_dist_false']].min(axis = 1)
    df_out['T_F_Out'] = df_out.apply(lambda x: tf(x), axis = 1)

    return df_out


@st.cache
def load_model(): 
    predictor = ocr_predictor(pretrained=True)
    return predictor 

reader = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    # Save uploaded file to 'F:/tmp' folder.
    save_folder = 'F:/tmp'
    save_path = Path(save_folder, File.name)
    with open(save_path, mode='wb') as w:
        w.write(File.getvalue())

    with st.spinner("🤖 AI is at Work! "):
        doc = DocumentFile.from_images(image)
        result = reader(doc)
        print('*********')
        print(result)
        print('*********')
        # st.image(result.show()) #display image
        json_output = result.export()

        with open('result.json', 'w') as fp:
            json.dump(json_output, fp)

        true_false_with_sn = json_out(path_json_output = 'result.json')
        df_out = df_output(post_processed_output = true_false_with_sn)
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made with ❤️ by Team: IIT Jodhpur")





