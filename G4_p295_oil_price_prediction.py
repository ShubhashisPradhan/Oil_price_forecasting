
import pickle
import streamlit as st
import pandas as pd
import prophet
import matplotlib.pyplot as plt
from prophet.serialize import model_to_json, model_from_json

# loading the trained model

#pickle_in = open('prophet_model.pkl', 'rb')
#pred_price = pickle.load(pickle_in)
with open('serialized_model.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model



@st.cache_data()

def prediction(date,price,Volume,High,Low):
    predictions=0
    df=pd.DataFrame({'ds':pd.to_datetime(date),'price':[price]	,'Volume':[Volume],'High':[High],	'Low':[Low]})
    predictions=m.predict(df)
    pred=predictions[['ds','yhat','yhat_lower','yhat_upper']]
    return pred.yhat[0]

# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """           OIL PRICE PREDICTION             """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)

    # following lines create boxes in which user can enter data required to make prediction
    date = st.sidebar.date_input("Choose a start date", pd.to_datetime('2020-01-01'))
    price = st.number_input("last price (last trading price) :")
    Volume = st.number_input("volume of oil expected to be trade :")
    High = st.number_input("price (upper limit of yester day) :")
    Low = st.number_input("price (lower limit of yester day) :")

    result =""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result= prediction(date,price,Volume,High,Low)
        st.success('{}'.format(result))

    loaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if loaded_file is not None:
      data = pd.read_csv(loaded_file)
      st.write("Uploaded Data:")
      st.write(data)
      temp=data.copy()
      data=data.iloc[:,[0,2,3,4,5]]
      data=data.rename({'Date':'ds','Open':'price'},axis=1)
      #data.head()

      #plotting
      st.title("Data Visualization")
      fig, ax = plt.subplots()
      data.drop(['ds'],axis=1).plot(subplots=True,ax=ax)
      st.pyplot(fig)

      #st.write()

      forecast1=m.predict(data)
      data['predicted']=forecast1[['yhat']]
      #forecast1
      st.write(data)
      fig, axes = plt.subplots(figsize=(16, 6))
      axes.plot(temp['ds'],temp['y'],color='red',label='predicted')
      axes.plot(data['ds'],data['predicted'],color='blue',label='actual')
      axes.legend()
      st.pyplot(fig)

if __name__=='__main__':
    main()
