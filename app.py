import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

ratings_df = pd.read_csv("D:/School Related Documents and Apps/4th Year/Machine Learning/books-host/book_recommender_model_deployment/dataset/ratings.csv")
books_df = pd.read_csv("D:/School Related Documents and Apps/4th Year/Machine Learning/books-host/book_recommender_model_deployment/dataset/books.csv")

st.title("The Email Classification Predictor")
st.subheader("Determine if that email you received is spam or not")
html_temp = """
	<div style="background-color:black;padding:10px">
	<h3 style="color:white;text-align:center;">Machine Learning</h3>
	</div>
	<div>
	<h3 style="color:black;text-align:center;">Done By </h3>
	<p style="color:black;text-align:center;">Deogratias Amani</p>
	
	"""
st.markdown(html_temp,unsafe_allow_html=True)

# st.dataframe(ratings_df)

user_input =st.text_area("Copy your email here","1,9999")
st.write("***testing;***\n\n" , user_input)

b_id =list(ratings_df.book_id.unique())
b_id.remove(10000)
book_arr = np.array(b_id)
user = np.array([53424 for i in range(len(b_id))])

st.write(len(user))
st.write(len(b_id))
# 
#model_file = "model.h5"
model = load_model( 'model.h5' )
#model = load_model(model)

pred = model.predict([book_arr, user])
pred

pred = pred.reshape(-1)
pred_ids = (-pred).argsort()[0:5]
pred_ids

if st.button("Predict"):
	
	st.write(books_df.iloc[pred_ids])
    # if y_pred[0] == 0:
    #     st.write("test1")
    # elif y_pred[0] == 1:
    #     st.write("test2")
