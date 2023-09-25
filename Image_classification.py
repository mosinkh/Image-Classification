import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import scipy
import shutil
from scipy import signal
from scipy import misc
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Image Classification", layout="wide")


model = load_model('Imgae_classification_1.h5')

Tiltle = "<p style='text-align: center;font-size: 44px;color: #000000;font-weight: bold;font-family: Cambaria;'>Image Classification</p>"
st.markdown(Tiltle, unsafe_allow_html=True)



select_option = st.sidebar.radio('Select an option',["Single Image","Folder"])
if select_option == "Single Image":
    st.sidebar.title("Single Image Selections.")
    upload_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if upload_image is not None:
        t_image = Image.open(upload_image)
        # custom_width = 150 
        # custom_height = 150 

        col0,col1, col2 = st.columns([0.5,3,3])
        col1.image(t_image, caption='Uploaded Image', width=150, use_column_width=False)

        if st.sidebar.button("Predict"):
            test_image = image.load_img(upload_image, target_size = (128, 128, 3))
            test_image = image.img_to_array(test_image)
            test_image=test_image/255
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)

            # custom_figsize = (20,20)
            # fig, ax = plt.subplots(figsize=custom_figsize)
            # ax.imshow(test_image)
            # test_width = 300
            # test_height = 300

            col2.image(test_image, width=300, use_column_width=False)
            # plt.figure(figsize=(5,5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(t_image)

            if result < 0:
                title = ("<p style='color:red; font-weight: bold;font-family: Arial;'>Unusable Image</p>")
                col2.markdown(title, unsafe_allow_html=True)

            else:
                title = ("<p style='color:green;font-weight: bold; font-family: Helvetica;'>Usable Image</p>")
                col2.markdown(title, unsafe_allow_html=True)
            

elif select_option == "Folder":
    results_df = pd.DataFrame(columns=["Image Name", "Status"])
    st.sidebar.title("Multiple Images Selection")
    folder_path = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if folder_path:
        if st.sidebar.button("Predict All"):
            for sr_no, uploaded_file in enumerate(folder_path, start=1):
                for uploaded_file in folder_path:
                    # Check if the file is an image
                    if uploaded_file.type.startswith('image/'):
                        image_name = uploaded_file.name
                        image_path = Image.open(uploaded_file)
                        col0,col1, col2,col3 = st.columns([0.5,3,3,1])
                        col1.image(image_path, width=250, use_column_width=False, caption=uploaded_file.name)
                        test_image = image.load_img(uploaded_file, target_size=(128, 128,3))
                        test_image = image.img_to_array(test_image)
                        test_image = test_image / 255.0
                        test_image = np.expand_dims(test_image, axis=0)
                        result = model.predict(test_image)
                        t_image = Image.open(uploaded_file)

                        col2.image(t_image, width=250, use_column_width=False)
                        
                        if result < 0:
                            status = "Unusable"
                            title = ("<p style='color:red; font-weight: bold;font-family: Arial;'>Unusable Image</p>")
                            col2.markdown(title, unsafe_allow_html=True)
                            unusable_folder = "D:/unusable_images"
                            os.makedirs(unusable_folder, exist_ok=True)
                            unusable_image_path = os.path.join(unusable_folder, uploaded_file.name)
                            uploaded_file.seek(0)  # Reset file pointer
                            with open(unusable_image_path, 'wb') as f:
                                f.write(uploaded_file.read())
                        else:
                            status = "Usable"
                            title = ("<p style='color:green;font-weight: bold; font-family: Helvetica;'>Usable Image</p>")
                            col2.markdown(title, unsafe_allow_html=True)
                            usable_folder = "D:/usable_images"
                            os.makedirs(usable_folder, exist_ok=True)
                            usable_image_path = os.path.join(usable_folder, uploaded_file.name)
                            uploaded_file.seek(0)  # Reset file pointer
                            with open(usable_image_path, 'wb') as f:
                                f.write(uploaded_file.read())

                        results_df = results_df.append({"Image Name": image_name, "Status": status}, ignore_index=True)
    
                break

        st.subheader("Final Result")
        st.write(results_df)
        
        if not results_df.empty:
            excel_data = io.BytesIO()
            with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
                results_df.to_excel(writer, sheet_name="Sheet1", index=False)
            # excel_data.seek(0)
            st.download_button(
                label="Download",
                data=excel_data,
                file_name="result.xlsx",
                key="download-button",
            )




