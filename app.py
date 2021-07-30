# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import datetime 
import pandas as pd
#import talib
#import ta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import requests
yf.pdr_override()

TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecasting - with fbprophet')

st.sidebar.header('User Input Parameters')
today = datetime.date.today()
def user_input_features():
    ticker = st.sidebar.text_input("Stock Symbol", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2018-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    est_years = st.sidebar.slider("Years of Prediction", 1, 4)
    return ticker, start_date, end_date, est_years

symbol, start, end, npredict = user_input_features()

selected_stock = symbol

#n_years = st.slider('Years of prediction:', 1, 4)
n_years = npredict
period = n_years * 365

@st.cache(allow_output_mutation=True)
def load_data(ticker):
    sdata = yf.download(ticker, start, TODAY)
    sdata.reset_index(inplace=True)
    return sdata

def load_name(ticker):
    detail = yf.Ticker(ticker)
    result = detail.info['longName']
    return result

#readme checkbox start
readme = st.checkbox("readme for details")

if readme:

    st.write("""
        This is a web app demo using [streamlit](https://streamlit.io/) library. It is hosted in [streamlit share](https://share.streamlit.io/blueapple16/stocks/main/app.py). You may get the codes via [github](https://github.com/blueapple16/stocks)
        """)


    st.write("""
        On the side bar, please enter the Stock Symbol and years of prediction on the stock pricings.
        """)


    st.write("""
Examples of Stock Symbol (any under [Yahoo Finance] (https://finance.yahoo.com/)) would be as below:
        """)

    st.write(pd.DataFrame({'Stock Symbol': ['5099.kl', 'AAPL', 'TSLA'],
'Details': ['Airasia BHD', 'Apple Inc', 'Tesla', ],
    }))


    st.write ("For more info, please contact:")

    st.write("<a href='https://www.linkedin.com/in/kah-wee-lim-02836a76/'>Kah Wee</a>",   unsafe_allow_html=True)
#readme checkbox end

st.markdown("""---""")

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... Retrieving details for '+ str(selected_stock))

company_name = load_name(selected_stock)

def background(target):
    bg = yf.Ticker(target)
    #    xxx = bg.info['xxx']
    #     bsector = bg.info['sector']
    if 'longBusinessSummary' in bg.info:
        blong = bg.info['longBusinessSummary']
    else:
        blong = 'Background not retrievable from Yahoo Finance'
    return blong

bglong = background(selected_stock)

st.subheader('Background of ' + str(company_name))
if selected_stock == '5099.kl':
    st.write(company_name + ' provides air transportation throughout Asia. '
                            'The airline operator focuses on delivering lower fares without a host of other amenities. '
                            'It does not provide frequent-flyer miles or airport lounges, but looks to cater affordable '
                            'transportation to all customers. In-flight meals and drinks are additional purchases available '
                            'to customers. All short and long-haul flights are nonstop, and the company focuses on high frequency '
                            'and high turnaround of flights. Operating segments are grouped by geographic regions. '
                            'Revenue derived from Malaysia makes up the majority of revenue, but the company does hold material '
                            'operations in several Asian regions.')

else:
    st.write(bglong)

st.subheader('Raw data of ' + str(company_name))
readme2 = st.checkbox("Tick to see last 5 days' Open/Close Performance")

if readme2:
    st.write(data.tail())

# testinfo = yf.Ticker(selected_stock)
# st.write(testinfo.info)
# st.write(testinfo.info['longBusinessSummary'])

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
	fig.layout.update(title_text='<b>Stock Chart </b>of ' + str(company_name) + '<br>Rangeslider for Date Adjustment', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
readme3 = st.checkbox("Tick to see last 5 Predicted days' Open/Close Performance")
if readme3:
    st.write(forecast.tail())


def plot_predict_data():
    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(title_text='<b>Stockprice Chart with Prediction</b> of ' + str(company_name) + '<br>Rangeslider for Date Adjustment', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

plot_predict_data()


st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

### Measure RME and R2 of fbprophet
st.subheader("R2, Mean Squared Error and Mean Absolute Error of the Prediction Modal")
metric_df = forecast.set_index('ds')[['yhat']].join(df_train.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)
r2 = r2_score(metric_df.y, metric_df.yhat)
st.write('The r2 score is '+ str(r2))
mse = mean_squared_error(metric_df.y, metric_df.yhat)
st.write('The mean squared error is '+ str(mse))
mae = mean_absolute_error(metric_df.y, metric_df.yhat)
st.write('The mean absolute error is '+ str(mae))

#st.subheader("Cross Validation of the Prediction Modal")
#from prophet.diagnostics import cross_validation
#df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
# ###st.write(df_cv.head())
#from prophet.plot import plot_cross_validation_metric
#fig4 = plot_cross_validation_metric(df_cv, metric='mape')
#st.write(fig4)


# from prophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)
# st.write(df_p.head())






# ### Technical Analysis of the stock
# # ### install, for streamlit share
# import streamlit as st
# import requests
# import os
# import sys
# import subprocess
# 
# # check if the library folder already exists, to avoid building everytime you load the pahe
# if not os.path.isdir("/tmp/ta-lib"):
#     # Download ta-lib to disk
#     with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
#         response = requests.get(
#             "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
#         )
#         file.write(response.content)
#     # get our current dir, to configure it back again. Just house keeping
#     default_cwd = os.getcwd()
#     os.chdir("/tmp")
#     # untar
#     os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
#     os.chdir("/tmp/ta-lib")
#     os.system("ls -la /app/equity/")
#     # build
#     os.system("./configure --prefix=/home/appuser")
#     os.system("make")
#     # install
#     os.system("make install")
#     # back to the cwd
#     os.chdir(default_cwd)
#     sys.stdout.flush()
# 
# # add the library to our current environment
# from ctypes import *
# 
# lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# # import library
# try:
#     import talib
# except ImportError:
#     subprocess.check_call(
#         [sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/",
#          "--global-option=-I/home/appuser/include/", "ta-lib"])
# finally:
#     import talib
# 
# # ### Start TA codes
# st.markdown("""---""")
# TA_Avail = ['SMA & EMA', 'Bollinger Band', 'RSI']
# TA_Select = st.sidebar.multiselect('Select Technical Analysis (TA) to be performed', TA_Avail)
# st.write("""
# Please select a Technical Analysis option from the sidebar to show relevant charts
# """)
# 
# # ## SMA and EMA
# # Simple Moving Average
# data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
# 
# # Exponential Moving Average
# data['EMA'] = talib.EMA(data['Close'], timeperiod=20)
# 
# 
# # Plot
# def plot_ta1_data():
#     figta1 = go.Figure()
#     figta1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
#     figta1.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], name="SMA 20"))
#     figta1.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], name="EMA 20"))
#     figta1.layout.update(
#         title_text='<b>SMA 20 vs EMA 20 </b>of ' + str(company_name) + '<br>Rangeslider for Date Adjustment',
#         xaxis_rangeslider_visible=True)
#     st.plotly_chart(figta1)
# 
# 
# # Bollinger Bands
# data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Close'], timeperiod=20)
# 
# 
# # Plot
# def plot_ta2_data():
#     figta2 = go.Figure()
#     figta2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
#     figta2.add_trace(go.Scatter(x=data['Date'], y=data['upper_band'], name="Upper Band"))
#     figta2.add_trace(go.Scatter(x=data['Date'], y=data['middle_band'], name="Middle Band"))
#     figta2.add_trace(go.Scatter(x=data['Date'], y=data['lower_band'], name="Lower Band"))
#     figta2.layout.update(
#         title_text='<b>Bollinger Bands </b>of ' + str(company_name) + '<br>Rangeslider for Date Adjustment',
#         xaxis_rangeslider_visible=True)
#     st.plotly_chart(figta2)
# 
# 
# # ## RSI (Relative Strength Index)
# # RSI
# data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
# 
# 
# # Plot
# def plot_ta3_data():
#     figta3 = go.Figure()
#     figta3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
#     figta3.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name="RSI 14"))
#     figta3.layout.update(title_text='<b>RSI 14 </b>of ' + str(company_name) + '<br>Rangeslider for Date Adjustment',
#                          xaxis_rangeslider_visible=True)
#     st.plotly_chart(figta3)
# 
# 
# # Plot for multiselect
# for i in range(len(TA_Avail)):
#     if (TA_Avail[i]) in TA_Select and i == 0:
#         plot_ta1_data()
#     elif (TA_Avail[i]) in TA_Select and i == 1:
#         plot_ta2_data()
#     elif (TA_Avail[i]) in TA_Select and i == 2:
#         plot_ta3_data()


