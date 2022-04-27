import pickle

from numpy import empty
import streamlit as st
import re
import string
import pandas as pd
filename = 'voting_model_nlp_final.sav'
loaded_model = pickle.load(open(filename, 'rb'))
st.title("Language Detection Hinglish")
user_input=st.text_area("Enter Text")
punc=dict((ord(char),None) for char in string.punctuation)
text=user_input.lower()
text=re.sub(r"\d+","",text)
text=text.translate(punc)
text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
list_pred=[]
clean_text=text.split()
for i in clean_text:
    prediction=loaded_model.predict([i])
    list_pred.append(prediction[0])

data= pd.DataFrame(list(zip(clean_text,list_pred)),columns =['Word','Prediction'])
if(not data.empty):
    #output=data.to_string(index=False)
    st.dataframe(data)