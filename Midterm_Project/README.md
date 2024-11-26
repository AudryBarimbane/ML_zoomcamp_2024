<h2>Predicting L'Oréal Stock Price Direction (CAC 40)</h2>
<h2>Problem Description</h2>

 <p>
   The goal is to predict if L'Oréal's stock price, which is part of the CAC 40 index, will go up or down the next trading day. This prediction uses technical indicators from past stock data, like moving averages, volatility, and RSI. 
 </p>
 <hr>
<h2>Dataset Description</h2>
<p>
 The data for this project includes historical stock market information for L'Oréal, a leading company in the CAC 40 index.
</p>
<h3>features</h3>
<ul>
 <li>Date : dataset is organized by dates, representing daily trading sessions.</li>
 <li>Open: Stock price at the beginning of the trading session.</li>
 <li>High: Highest price during the session.</li>
 <li>Low: Lowest price during the session.</li>
 <li>Close: Stock price at the end of the session (used for predictions).</li>
 <li>Volume: Total number of shares traded during the session.</li>
 <li>MA5, MA10, and MA30 represent the average closing prices over 5, 10, and 30 days, respectively.</li>
 <li>Log return : log difference between consecutive closing prices to measure relative price changes</li>
 <li>Volatility : Rolling standard deviation of logarithmic returns over 30 days, scaled to a monthly level, to measure price variability.
</li>
 <li>Relative Strength Index (RSI) : Measures the momentum of price changes over a 14-day period to identify overbought or oversold conditions.</li>
 
</ul>
<h3>Target Variable</h3>
   <ul>
    <li>1 if the stock price increases the next day.</li>
    <li>0 if the stock price decreases the next day.</li>
   </ul>

<hr>
<h2>Project Description</h2>
<p>The primary goal of this project is to develop a machine learning model that can accurately predict whether L'Oréal’s stock price will increase or decrease on the following trading day, based on past price data and technical analysis</p>
<h3>Exploratory Data Analysis (EDA)</h3>
<ul>
<li>Data Collection: The script uses the yfinance library to download L'Oréal's historical stock data (ticker: OR.PA) from 2020-01-01</li>
 <li>Missing values are checked and handled by filling them with zeros</li>
</ul>
<hr>

<h2>Files</h2>
<hr>
<h2>Run the code</h2>
<hr>
<h2>Docker</h2>
