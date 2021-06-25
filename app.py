import os
import streamlit as st
from tensorflow import keras
from db import Image
from PIL import Image as PI
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from skimage.transform import rescale, resize
import tensorflow as tf
import skimage.io
import numpy as np
import warnings # tf needs to learn to stfu
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

st.title("AI based Skin Disease classification")
st.subheader('final yr project')

target_names = [('akiec', 'Actinic keratoses and intraepithelial carcinomae'), 
                ('bcc', ' basal cell carcinoma'), 
                ('bkl', 'benign keratosis-like lesions'), 
                ('df', 'dermatofibroma'), 
                ('nv', ' melanocytic nevi'), 
                ('vasc', ' pyogenic granulomas and hemorrhage'), 
                ('mel', 'melanoma')]

def load_model():
    MODEL_PATH = r"ai\model"
    return  keras.models.load_model(MODEL_PATH)

def resize_img(img):
    resized_image = resize(img, (28, 28))
     # Convert the image to a 0-255 scale.
    rescaled_image = 255 * resized_image
     # Convert to integer data type pixels.
    final_image = rescaled_image.astype(np.uint8)
     # show resized image
    img = PI.fromarray(final_image, 'RGB')
    test_img  = np.array(img)
    return test_img.reshape(-1, 28, 28, 3)

def test_img_from_url(model, url):
    image_np = skimage.io.imread(url)
    test = resize_img(image_np)
    result = model.predict(test).round()
    return target_names[np.argmax(result)]

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_file(file,path):
    try:
        db = opendb()
        ext = file.type.split('/')[1] # second piece
        img = Image(filename=file.name,extension=ext,filepath=path)
        db.add(img)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

choice = st.sidebar.selectbox("select option",['view uploads','upload content','manage uploads'])

if choice == 'upload content':
    file = st.file_uploader("select a image",type=['jpg','png'])
    if file:
        path = os.path.join('uploads',file.name)
        with open(path,'wb') as f:
            f.write(file.getbuffer())
            status = save_file(file,path)
            if status:
                st.sidebar.success("file uploaded")
                try:st.sidebar.image(path,use_column_width=True)
                except:st.error("image could not be loaded")
            else:
                st.sidebar.error('upload failed')

if choice == 'view uploads':
    db = opendb()
    results = db.query(Image).all()
    db.close()
    img = st.sidebar.radio('select image',results)
    if img and os.path.exists(img.filepath):
        st.sidebar.info("selected img")
        try:
            st.image(img.filepath, use_column_width=True)
        except:st.error("please try another image")
        if st.button("analyse"):
            model = load_model()
            out = test_img_from_url(model,img.filepath)
            st.title("Prediction complete")
            st.balloons()
            st.markdown(f"""
            ## Skin Disease predicted is 
            # disease : {out[1].split()[-1]}
            or
            ## medical name :
            {out[1]}{out[0]}
            """)
            
        
if choice == 'manage uploads':
    db = opendb()
    # results = db.query(Image).filter(Image.uploader == 'admin') if u want to use where query
    results = db.query(Image).all()
    db.close()
    img = st.sidebar.radio('select image to remove',results)
    if img:
        st.error("img to be deleted")
        if os.path.exists(img.filepath):
            try:st.sidebar.image(path,use_column_width=True)
            except:st.error("image could not be loaded")
        if st.sidebar.button("delete"): 
            try:
                db = opendb()
                db.query(Image).filter(Image.id == img.id).delete()
                if os.path.exists(img.filepath):
                    os.unlink(img.filepath)
                db.commit()
                db.close()
                st.info("image deleted")
            except Exception as e:
                st.error("image not deleted")
                st.error(e)
            

