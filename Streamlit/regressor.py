import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PIL import Image

st.title("Linear Regression Model")

uploaded_file = st.file_uploader("Upload the Dataset keeping the target variable column at the end")
image = Image.open('LinReg.png')
st.image(image)
#st.write(uploaded_file.getvalue())
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    with st.sidebar:
      value = st.slider("Select the test size",0.1,0.9)
      rs = st.slider("Select the random state",0,100)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=value,random_state=rs)
    # st.write(df.shape)
    # st.write(x_train.shape)
    # st.write(y_train.shape)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    r = round(r2_score(y_test, y_pred),3)
    st.write("The model is ready with the R-squared value ",r)
    columns = df.columns
    x_input=[]
    input=[]
    for c in range(len(columns)-1):
        n = st.number_input(columns[c])
        input.append(n)
    x_input.append(input)
    y_output = regressor.predict(x_input)
    st.write("The Model Output: ")
    st.write(f"Predicted {columns[-1]}: ",round(*y_output,2))





