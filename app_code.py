# -*- coding: utf-8 -*-
import streamlit as st
import os
from PIL import Image
try:
    import cv2
except:
    !pip install cv2
    import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    
#Creamos una funcion para mapear el animal 
def dog_cat_mapping(a):
    if a=="dogs":
        return 1
    else:return 0

#Creamos una funcion para fenerar el dataset con las imagenes de una carpeta
def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
    
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            try:
                IMG_HEIGHT=200
                IMG_WIDTH=200
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            except:
                break
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name



#Creamos una funcion para crar y entranar el modelo que usaremos, como en el repo original
def model(input_shape):
    model=Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape,activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(2,2),activation='relu'))
    model.add(Conv2D(128, kernel_size=(2,2),activation='relu'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    return model

def format_img_to_model(img_cv2):
    # Redimensionar la imagen a 200x200 píxeles
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    img_cv2_formatted = cv2.resize(img_cv2, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    # Normalizar la imagen
    img_cv2_formatted = img_cv2_formatted.astype('float32')
    img_cv2_formatted /= 255
    return img_cv2_formatted

#Creamos una funcion para redimensionar las imagenes, para los laterales
def img_resize(img_path):
    img_object= Image.open(img_path)
    img_re_re = img_object.resize((200,200))
    return(img_re_re)
    
#Creamos una funcion para guardar el codigo propio del front de la 
#app con streamlit
def streamlit_app():
    #Metemos los logotipos de la universidad en nuestra app    
    #Recuperamos la ruta donde esta este codigo
    script_directory = os.path.dirname(os.path.abspath(__file__))

    #Construir las rutas de las imagenes
    logo_escuela_negocios = os.path.join(script_directory, 'img', 'logo_escuela_negocios.jpeg')
    logo_uemc = os.path.join(script_directory, 'img', 'logo_uemc.jpg')

    #Recuperamos las imagenes con PIL.Image
    logo_esne = Image.open(logo_escuela_negocios)
    logo_uemc= Image.open(logo_uemc)


    #Redimensionamos las imagenes
    logo_uemc_re = logo_uemc.resize((logo_uemc.width,int(logo_uemc.height*1.4)))

    #Dividimos el espacio en 4 y usamos los extremos para que quede mas estetico
    col1, col2, col3= st.columns(3)
    with col1:
        st.image(logo_esne, use_column_width=True)
    with col3:
        st.image(logo_uemc_re, use_column_width=True)

    #Creamos un titulo para la aplicacion, lo hacemos con markdown para centrarlo
    st.markdown("<h1 style='text-align: center;'>Técnicas de desarrollo avanzado de aplicaciones BigData</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>catdog con streamlit</h2>", unsafe_allow_html=True)

   
    #Ahora que tenemos la "cabecera" de la app empezamos con el cuerpo
    #Dividiremos el espacio en 10, en las dos columnas externas pondremos imagenes de
    #perros y gatos, en las 6 centrales pondremos el cuerpo de la app
    #un espacio para cargar nuestras imagenes y obtener los resultados del modelo
    
    #Cargamos las imagenes de los laterales
    img_dog_1 = img_resize(os.path.join(script_directory, 'img', 'dog1.jpg'))
    img_dog_2 = img_resize(os.path.join(script_directory, 'img', 'dog2.jpg'))
    img_dog_3 = img_resize(os.path.join(script_directory, 'img', 'dog3.jpg'))
    img_dog_4 = img_resize(os.path.join(script_directory, 'img', 'dog4.jpg'))
    img_cat_1 = img_resize(os.path.join(script_directory, 'img', 'cat1.jpg'))
    img_cat_2 = img_resize(os.path.join(script_directory, 'img', 'cat2.jpg'))
    img_cat_3 = img_resize(os.path.join(script_directory, 'img', 'cat3.jpg'))
    img_cat_4 = img_resize(os.path.join(script_directory, 'img', 'cat4.jpg'))
    
    # Dividir el espacio en 10 columnas
    col_l1, col_l2, col_central, col_r1, col_r2 = st.columns([1,1,6,1,1])
    
    # En las dos primeras columnas, colocar varias imágenes una debajo de otra
    with col_l1:
        st.image(img_dog_1, use_column_width=True)
        st.image(img_cat_1, use_column_width=True)
        st.image(img_dog_4, use_column_width=True)
        st.image(img_cat_4, use_column_width=True) 
    
    with col_l2:
        st.image(img_cat_2, use_column_width=True)
        st.image(img_dog_2, use_column_width=True)
        st.image(img_cat_3, use_column_width=True)
        st.image(img_dog_3, use_column_width=True)

    with col_r1:
        st.image(img_cat_3, use_column_width=True)
        st.image(img_dog_3, use_column_width=True)
        st.image(img_cat_2, use_column_width=True)
        st.image(img_dog_2, use_column_width=True)
    
    with col_r2:
        st.image(img_dog_4, use_column_width=True)
        st.image(img_cat_4, use_column_width=True)    
        st.image(img_dog_1, use_column_width=True)
        st.image(img_cat_1, use_column_width=True)

    
    # En las columnas centrales, colocamos el "cuerpo" de la app
    with col_central:
        #Creamos una rectangulo para cargar una imagen
        uploaded_image = st.file_uploader("Subir imagen")
    
        #Hacemos un boton "Try yourself"
        if st.button("Try yourself"):
            if uploaded_image is not None:
                #Procesamos la imagen para llevarla al formato adecuado para el predict
                image_cv2 = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
                
                formatted_img=format_img_to_model(image_cv2)
                
                prediction=model.predict(formatted_img)
                predict_probs=model.predict_proba(formatted_img)
                
                if prediction < 0.5:
                     st.write(f"Se trata de un gato, hay una probabilidad de {predict_probs[0][0]} de ello.")
                else:
                    st.write(f"Se trata de un perro, hay una probabilidad de {predict_probs[0][1]} de ello.")
                    

if __name__ == '__main__':
    #Recuperamos el directorio donde se ejecuta este script
    #en este directorio estan todos nuestros recursos
    script_directory = os.path.dirname(os.path.abspath(__file__))

    #extremos los datos, como en el repo
    img_data, class_name =create_dataset(os.path.join(script_directory,'training_set'))

    #pasamos estas variables a matrices numpy para su posterior tratamiento por el modelo
    img_data=np.array(img_data)
    class_name=np.array(class_name)

    #Redefinimos la clase(gato-perro)    
    class_name=list(map(dog_cat_mapping,class_name))
    class_name=np.array(class_name)

    #Creamos el modelo, lo guardamos y lo entrenamos
    input_shape=img_data.shape[1:]
    model=model(input_shape)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    #Entrenamos el modelo
    model.fit(x=img_data,y=class_name, epochs=10)
    
    #Ejecutamos una vez tenemos el modelo entrenado ejecutamos
    #el codigo propio de la aplicacion
    streamlit_app(model)
    
