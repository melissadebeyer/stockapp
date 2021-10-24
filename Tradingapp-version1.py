#!/usr/bin/env python
# coding: utf-8

# In[69]:


import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import requests
import talib 
import ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
yf.pdr_override()
import altair as alt
from yahoofinancials import YahooFinancials
import requests
from PIL import Image


# In[34]:


stockss = ['VPK.AS','UNA.AS','HEIA.AS','ADYEN.AS']


# In[46]:


stocks=['ADYEN.AS','MT.AS','ASML.AS','AH.AS','AGN.AS','RDSA.AS','DSM.AS','VWA.AS','SIA.AS','RDSB.AS','PBD.AS','MEURV.AS','HUNDP.AS','TNTE.AS','UL.AS','TFG.AS','INGA.AS','PHARM.AS','GTO.AS','HEIA.AS','YATRA.AS','ATRS.AS','KPN.AS','WKL.AS','KTC.AS','VPK.AS','BAMNB.AS','KENDR.AS','PHIA.AS','UNA.AS','OCI.AS','USG.AS','UNIA.AS','SBMO.AS','NSI.AS','IM.AS','CRBN.AS','BOKA.AS','ATC.AS','ARCAD.AS','AKZA.AS','WES.AS','TOM2.AS','SLIGR.AS','RAND.AS','NEWAY.AS','NEDSE.AS','ORANW.AS','AXS.AS','APAM.AS','LANS.AS','VALUE.AS','PREVA.AS','AALB.AS','ABN.AS','ACCEL.AS','ACOMO.AS','AMG.AS','AJAX.AS','AF.AS','DTEL.AS','ATCB.AS','AND.AS','HDG.AS','BBV.AS','BAMA.AS','ASM.AS','GEN.AS','ICT.AS','BATEN.AS','BBED.AS','GLPG.AS','FAGR.AS','BEVER.AS','BESI.AS','BINCK.AS','HAL.AS','BOLS.AS','BOEI.AS','BRNL.AS','BTGP.AS','IBMA.AS','CHTEX.AS','CIS.AS','CLB.AS','CTAC.AS','CURE.AS','SOURC.AS','DICO.AS','DL.AS','DOCD.AS','DPA.AS','ECMPA.AS','ECT.AS','REN.AS','ENI.AS','ESP.AS','TFA.AS','SANTA.AS','FLOW.AS','PORF.AS','SGO.AS','FUR.AS','SOPH.AS','RFRG.AS','VTA.AS','GVNV.AS','HEIO.AS','HYDRA.AS','MTY.AS','IEX.AS','IMCD.AS','TIT.AS','TISN.AS','KARD.AS','KA.AS','KDS.AS','TIE.AS','BRILL.AS','LVIDE.AS','MACIN.AS','MSF.AS','RBST7.AS','RBST6.AS','RBST5.AS','NSE.AS','VNC.AS','NEDAP.AS','BALNE.AS','ORDI.AS','HOLCO.AS','ROYRE.AS','UNCP7.AS','UNCC7.AS','NOVI.AS','GROHA.AS','STRN.AS','ROOD.AS','PNL.AS','RNS.AS','TWEKA.AS','SNOW.AS','UNCP6.AS','INVER.AS','NN.AS','TMG.AS','INTER.AS','OCPET.AS','RBS.AS','PEP.AS','TBIRD.AS','VERIZ.AS','E-ON.AS','INCO.AS']


# In[73]:


st.title("MNB's stock analysis app")
st.markdown("The graphs and tables below can be used to determine an optimal stock portfolio based on the Sharpe ratio and will give you detailed information on any given stock")
image = Image.open(r'C:\Users\melis\Pictures\wallstphoto.jpg')
st.image(image)

st.sidebar.header('User Input Parameters')

today = datetime.date.today()
def user_input_features():
    symbol = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2021-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    tickerlist=st.sidebar.multiselect('Which stocks would you like to include in the Sharpe ratio analysis?',stocks, ['ASML.AS','ADYEN.AS'])
    return symbol, start_date, end_date, tickerlist

symbol, start, end, tickerlist = user_input_features()


start1 = pd.to_datetime(start)
end1 = pd.to_datetime(end)


# In[28]:


#below is the script for sharpe optimization of Dutch stocks


# In[67]:


yahoo_financials = YahooFinancials(tickerlist)

data = yahoo_financials.get_historical_price_data(start, 
                                                  end, 
                                                  time_interval='daily')

prices_df = pd.DataFrame({
    a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in tickerlist
})


# In[58]:


returns =prices_df.pct_change()
meanDailyreturns=returns.mean()
daily_cumu_returns=(1+returns).cumprod()
#covariance matrix from daily returns
cov_matrix_d=(returns.cov())*252


# In[59]:


Sigma = risk_models.sample_cov(prices_df)
mu = expected_returns.mean_historical_return(prices_df)
ef=EfficientFrontier(mu, Sigma)


# In[60]:


weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()


# In[61]:


weightdf=pd.DataFrame(cleaned_weights.items(), columns=['Stock', 'Weight'])
weightdf= weightdf.set_index('Stock')
weightdf1=weightdf[weightdf['Weight']!=0]


# In[62]:


stats=(ef.portfolio_performance(verbose=True))


# In[76]:


# Plot
st.title("Optimized Sharpe ratio portfolio")
weightdf1

st.bar_chart(weightdf1['Weight'])


# In[64]:


st.header("Portfolio stats")
st.markdown("Below you can see the expected annual return (x100%), annual volatility (x100%) and the sharpe ratio")
st.write(ef.portfolio_performance(verbose=False))


# In[65]:


# below is the script for the analysis of individual stocks


# In[ ]:


st.title("Individual stock analysis")


# In[19]:


# Read data 
data = yf.download(symbol,start1,end1)

# Adjusted Close Price
st.header(f"Adjusted Close Price")
st.line_chart(data['Adj Close'])

# ## SMA and EMA
#Simple Moving Average
data['SMA'] = talib.SMA(data['Adj Close'], timeperiod = 20)

# Exponential Moving Average
data['EMA'] = talib.EMA(data['Adj Close'], timeperiod = 20)

# Plot
st.header(f"Simple Moving Average vs. Exponential Moving Average")
st.line_chart(data[['Adj Close','SMA','EMA']])

# Bollinger Bands
data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)

# Plot
st.header(f"Bollinger Bands")
st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

# ## MACD (Moving Average Convergence Divergence)
# MACD
data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Plot
st.header(f"Moving Average Convergence Divergence")
st.line_chart(data[['macd','macdsignal']])

## CCI (Commodity Channel Index)
# CCI
cci = ta.trend.cci(data['High'], data['Low'], data['Close'])

# Plot
st.header(f"Commodity Channel Index")
st.line_chart(cci)

# ## RSI (Relative Strength Index)
# RSI
data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

# Plot
st.header(f"Relative Strength Index")
st.line_chart(data['RSI'])

# ## OBV (On Balance Volume)
# OBV
data['OBV'] = talib.OBV(data['Adj Close'], data['Volume'])/10**6

# Plot
st.header(f"On Balance Volume")
st.line_chart(data['OBV'])


# In[ ]:


#BELOW IS THE NEWS SENTIMENT ANALYSIS

