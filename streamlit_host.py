import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("my_model_new_new_using_for_loop_final.hdf5")
### load file
st.title("Animal Prediction using :green[CNN VGG16] Transfer Learning Architecture")
st.markdown("Upload the image of the animlas :green[[Leopard, Lion, Elephant, Wolf, Bear]]")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(150,150))
    resized_tensor = resized.reshape((1,150,150,3))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    # resized = mobilenet_v2_preprocess_input(resized)
    # img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(resized_tensor)
        if(prediction[0][0] == 1):
            st.markdown("Predicted animal in the image can be :green[Bear]")
            st.success("The Accuracy of the Deep Learning CNN VGG16 transfer learning model is : ")
            st.image('https://i.postimg.cc/XqWJwJM8/Accuracy.png')
            
        if(prediction[0][1] == 1):
            st.markdown("Predicted animal in the image can be :green[Elephant]")
            st.success("The Accuracy of the Deep Learning CNN VGG16 transfer learning model is : ")
            st.image('https://i.postimg.cc/XqWJwJM8/Accuracy.png')
        if(prediction[0][2] == 1):
            st.markdown("Predicted animal in the image can be :green[Leopard]")
            st.success("The Accuracy of the Deep Learning CNN VGG16 transfer learning model is : ")
            st.image('https://i.postimg.cc/XqWJwJM8/Accuracy.png')
        if(prediction[0][3] == 1):
            st.markdown("Predicted animal in the image can be :green[Lion]")
            st.success("The Accuracy of the Deep Learning CNN VGG16 transfer learning model is : ")
            st.image('https://i.postimg.cc/XqWJwJM8/Accuracy.png')
        if(prediction[0][4] == 1):
            st.markdown("Predicted animal in the image can be :green[Wolf]") 
            st.success("The Accuracy of the Deep Learning CNN VGG16 transfer learning model is : ")
            st.image('https://i.postimg.cc/XqWJwJM8/Accuracy.png')
            
        