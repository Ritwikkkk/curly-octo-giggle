import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import base64
import numpy as np

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )



def main():
    # Set the background image using the file path
    add_bg_from_local('rfbg_1.jpg')  
    side_bg = 'forrest.jpg'
    sidebar_bg(side_bg)
  
    # Your Streamlit app content goes here
    # col1, col2 = st.columns((4,1))
    # with col1:
    #     st.title('Random Forest Classifier')
    #     st.write('Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result.')
    # with col2:
    #     st.image(Image.open('LOGO.jpg'),width = 200)
    st.title('Random Forest Classifier')
    st.write('Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result.')
    uploaded_file = st.file_uploader('Upload the dataset by keeping the target variable in the last column')
    #st.image(Image.open('randomforest.jpg'))
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        le = LabelEncoder()
        for i in range(len(df.columns)):
          if df[df.columns[i]].dtype == 'object':
            df[df.columns[i]] = le.fit_transform(df[df.columns[i]])
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        flag1 = 0
        flag2 = 0
        with st.sidebar:
            tts = st.selectbox('Choose the train-test split parameters?',options=['No','Yes'])
            if tts == 'Yes':
              st.title('Select the train-test split parameters')
              st.write("")
              test_size = st.slider('Select the test size',0.1,0.9)
              rs = st.slider('Select the random state',0,100)
              x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=rs)
              flag2 = 1
            value = st.selectbox('Choose the hyperparameters?',options=['No','Yes'])
            if value == 'Yes':
              st.title('Select the hyperparameters')
              st.write("")
              n = st.slider('Select the No. of trees',1,1000)
              options = ['gini','entropy']
              c = st.selectbox('Select the Criterion',options)
              md = st.slider('Select the max depth',1,50)
              mss = st.slider('Select Min Samples Split',2,100)
              msl = st.slider('Select Min Samples Leaf',1,100)
              flag1 = 1
        st.markdown(f"Click :blue[Train the Model] button below to train the model with default parameters, or choose the parameters from the sidebar and then click the :blue[Train the Model] button")
        ip=st.button('Train the Model')
        if ip:
          if flag1 == 0 and flag2 ==0:          
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
            rfc = RandomForestClassifier()
            rfc.fit(x_train, y_train)
            y_pred = rfc.predict(x_test)
            if accuracy_score(y_test, y_pred) >= 0.8:
              st.markdown(f"### The Model Accuracy is: :green[{round(accuracy_score(y_test, y_pred),4)*100}%]")
            elif accuracy_score(y_test, y_pred) >= 0.6:
              st.markdown(f"### The Model Accuracy is: :orange[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            elif accuracy_score(y_test, y_pred) < 0.6:
              st.markdown(f"### The Model Accuracy is: :red[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            if y_pred is not None:
              cv = st.selectbox('Do you want to use cross-validation techniques to optimize the hyper parameters?',options=['No','Grid Search CV','Randomized Search CV'])

          elif flag1 == 1 and flag2 == 0:
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
            rfc = RandomForestClassifier(n_estimators=n, criterion=c,max_depth=md,min_samples_split=mss, min_samples_leaf=msl)
            rfc.fit(x_train, y_train)
            y_pred = rfc.predict(x_test)
            if accuracy_score(y_test, y_pred) >= 0.8:
              st.markdown(f"### The Model Accuracy is: :green[{round(accuracy_score(y_test, y_pred),4)*100}%]")
            elif accuracy_score(y_test, y_pred) >= 0.6:
              st.markdown(f"### The Model Accuracy is: :orange[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            elif accuracy_score(y_test, y_pred) < 0.6:
              st.markdown(f"### The Model Accuracy is: :red[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            if y_pred is not None:
              cv = st.selectbox('Do you want to use cross-validation techniques to optimize the hyper parameters?',options=['No','Grid Search CV','Randomized Search CV'])
            
          elif flag1 == 1 and flag2 ==1:
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=rs)
            rfc = RandomForestClassifier(n_estimators=n, criterion=c,max_depth=md,min_samples_split=mss, min_samples_leaf=msl)
            rfc.fit(x_train, y_train)
            y_pred = rfc.predict(x_test)
            if accuracy_score(y_test, y_pred) >= 0.8:
              st.markdown(f"### The Model Accuracy is: :green[{round(accuracy_score(y_test, y_pred),4)*100}%]")
            elif accuracy_score(y_test, y_pred) >= 0.6:
              st.markdown(f"### The Model Accuracy is: :orange[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            elif accuracy_score(y_test, y_pred) < 0.6:
              st.markdown(f"### The Model Accuracy is: :red[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            if y_pred is not None:
              cv = st.selectbox('Do you want to use cross-validation techniques to optimize the hyper parameters?',options=['No','Grid Search CV','Randomized Search CV'])
          else:
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=rs)
            rfc = RandomForestClassifier()
            rfc.fit(x_train, y_train)
            y_pred = rfc.predict(x_test)
            if accuracy_score(y_test, y_pred) >= 0.8:
              st.markdown(f"### The Model Accuracy is: :green[{round(accuracy_score(y_test, y_pred),4)*100}%]")
            elif accuracy_score(y_test, y_pred) >= 0.6:
              st.markdown(f"### The Model Accuracy is: :orange[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            elif accuracy_score(y_test, y_pred) < 0.6:
              st.markdown(f"### The Model Accuracy is: :red[{round(accuracy_score(y_test, y_pred),4)*100}%] :blue[(Need Improvement)]")
            if y_pred is not None:
              cv = st.selectbox('Do you want to use cross-validation techniques to optimize the hyper parameters?',options=['No','Grid Search CV','Randomized Search CV'])
        columns = df.columns
        df2 = df.truncate(after=0)   
        df_input = df2.drop(columns[-1], axis = 1)
        user_input=[]
        #st.write(columns[0])
        #st.write(df_input[columns[0]])
        for c in range(len(columns)-1):
            if df_input[df_input.columns[c]].dtype == 'int64':
              n = st.number_input(columns[c])
              user_input.append(n)
            else: 
              n = st.text_input(columns[c])
              user_input.append(n)         
        if len(user_input) == len(columns)-1:
          user_in = pd.DataFrame(data = [user_input], columns = df_input.columns)
          #x_input.append(input)
          #df_input.append([user_input])
          #st.write(df_input.tail())
          le = LabelEncoder()
          for i in range(len(user_in.columns)):
            if user_in[user_in.columns[i]].dtype == 'object':
              user_in[user_in.columns[i]] = le.fit_transform(user_in[user_in.columns[i]])
          y_output = rfc.predict(user_in)
          st.write("The Model Output: ")
          #if df[df.columns[-1]].dtype == 'object':
          st.write(y_output)
          st.write(f"Predicted {columns[-1]}: ",le.inverse_transform(y_output))
          #else:
            #st.write(f"Predicted {columns[-1]}: ",y_output)
          dfr_train = y_train.copy()
          for column in y_train.columns:
              le.fit(y_output[column])   # you fit the column before it was encoded here

          # now that python has the above encoding in its memory, we can ask it to reverse such 
          # encoding in the corresponding column having encoded values of the split dataset

              dfr_train[column] = le.inverse_transform(y_train[column])
          st.write(dfr_train)
if __name__ == "__main__":
    main()

