import streamlit as st 
import tensorflow_hub as hub
import tensorflow as tf 
import numpy as np 
import PIL
import matplotlib.pyplot as plt
import requests
import cv2

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
       assert tensor.shape[0] == 1
    tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(url):
    max_dim = 512
    img = requests.get(url).content
    img =  tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img





style_image = cv2.imread('2.jpeg')
content_image = cv2.imread('2.jpeg')


st.set_page_config(layout="wide")
st.title("Neural_Style_Transfer")

col1,col2,col3,col4 = st.columns(4)
 
with col1 :
    clink = st.text_input('URL',value='https://www.wolfram.com/language/12/machine-learning-for-images/assets.en/built-in-image-style-transfer/O_1.png')
    content_image = load_img(clink)    
    st.image(tensor_to_image(content_image),width = 300)
with col2:
    slink = st.text_input('URL',value = 'https://blog.paperspace.com/content/images/2018/07/altamirano.jpg')
    style_image = load_img(slink)
    st.image(tensor_to_image(style_image),width = 300)
with col4:    
    st.header('OUTPUT_IMAGE')
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    img = tensor_to_image(stylized_image)
    st.image(img,width = 300)

c1,c2 = st.columns(2)
with col3:
    if(st.button('Transfer')):
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        img = tensor_to_image(stylized_image)


# st.image(image=img)