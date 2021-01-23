from flask import Flask, redirect, url_for, render_template, Response, request
import requests
import datetime as dt
import math
import yfinance as finance_api
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pandas
from pandas.plotting import register_matplotlib_converters
import tweepy
from matplotlib import style
import matplotlib.dates as dates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import io
import random
import statistics

import twitter
app = Flask(__name__)

register_matplotlib_converters()

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
url_2 = "&apikey=T10BG6UPLFSRKTFU"

name_url = "https://api-v2.intrinio.com/companies/"
name_url_2 = "?api_key=OjkwYTJmMzc1YWU1OGE4MDgxZmQ2YjlmYjRjZTBiZmJi"

news_url = "https://newsapi.org/v2/everything?q="
news_url_2 = "&from=2021-01-09&to=2021-01-14&sortBy=popularity&apiKey=2ca74f9bf01f4f648650d30f0381f633"

consumer_key="dLWxOxvWyQHvjh97moMHL63tF"
consumer_secret="chasb9g6x9Y0UrCnjbPTV02FsGPXLkF6QkfNSR7Q1FsOxhjxES"
access_token="24473540-TqUHLlFh5n7uH4JBhWZbWulUyAfY1Rk93WhoctGHT"
access_token_secret= "LAEnZPHxM7VS0hNm4uhiG2dIEUwrVm4EnP3Vu3gbLsZQv"

daily_stock_all = None
company_name = None

def retrieve_data(ticker):
    stock_JSON = requests.get(url + ticker + url_2).json()
    daily_stock_all = stock_JSON['Time Series (Daily)']

    name_JSON = requests.get(name_url + ticker + name_url_2).json()

    news_JSON = requests.get(news_url + ticker + news_url_2).json()
    all_news = news_JSON['articles']

    return  [daily_stock_all, name_JSON, news_JSON]

ticker = 'AAPL'


@app.route('/')
def index():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['tickerText']
    ticker = text.upper()
    return home(ticker)

@app.route('/home')
def home(ticker):
    daily_stock_all, name_JSON, news_JSON = retrieve_data(ticker)
    company_name = name_JSON['name']
    all_news = news_JSON['articles']
    initial_plot = primary_stock_graph(daily_stock_all, company_name)
    min_stock, max_stock, average_stock = min_max_values(daily_stock_all, company_name)
    all_headline_titles, all_headline_links, all_headline_image_links, all_headline_authors, all_headline_descriptions,all_headline_publish_dates = retrieve_news_info(all_news)
    total_polarity, polarity_list, tweet_list, tweet_user_list, tweet_user_pic_list = detect_tweet_sentiment(ticker)
    final_tweet_recommendation_string = final_tweet_recommendation(total_polarity, company_name)
    news_polarity_list, total_news_polarity = detect_news_sentiment(all_news)
    final_news_recommendation_string = final_news_recommendation(total_news_polarity, company_name)
    return render_template("index.html", ticker = ticker, name_JSON = name_JSON, company_name = company_name, 
    min_stock = min_stock, max_stock = max_stock, average_stock = average_stock, all_headline_titles = all_headline_titles, 
    all_headline_links = all_headline_links, all_headline_image_links = all_headline_image_links, all_headline_authors = all_headline_authors,
    all_headline_descriptions = all_headline_descriptions, all_headline_publish_dates = all_headline_publish_dates, total_polarity = total_polarity,
    polarity_list = polarity_list, tweet_list = tweet_list, tweet_user_list = tweet_user_list, tweet_user_pic_list = tweet_user_pic_list,
    final_tweet_recommendation_string = final_tweet_recommendation_string, final_news_recommendation_string = final_news_recommendation_string,
    total_news_polarity = total_news_polarity)

@app.route('/initialplot.png')
def initial_plot_png():
    daily_stock_all, name_JSON, news_JSON = retrieve_data(ticker)
    company_name = name_JSON['name']
    fig = primary_stock_graph(daily_stock_all, company_name)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/tweetplot.png')
def tweet_plot_png():
    total_polarity, polarity_list, tweet_list, tweet_user_list, tweet_user_pic_list = detect_tweet_sentiment(ticker)
    fig = tweet_graph(polarity_list)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/newsplot.png')
def news_plot_png():
    daily_stock_all, name_JSON, news_JSON = retrieve_data(ticker)
    all_news = news_JSON['articles']
    news_polarity_list, total_news_polarity = detect_news_sentiment(all_news)
    fig = tweet_graph(news_polarity_list)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/competitorplot.png')
def competitor_plot_png():
    ticker_list = ['AAPL']
    fig = retrieve_competitor_bar_chart(ticker_list)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def primary_stock_graph(daily_stock_all, company_name):
    close_value_list = []
    count_list = []
    days = []
    count = 100
    for key, value in daily_stock_all.items():
        closing_price = float(value['4. close'])
        close_value_list.append(closing_price)
        count_list.append(count)
        days.append(key)
        count = count - 1
    plt.clf()
    fig = Figure()
    fig.set_size_inches(10.5, 7.5)
    fig.clear()

    axes = fig.add_subplot(1,1,1)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Value ($)', fontsize=16)
    x_days = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in days]
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=10))
    axes.plot(x_days, (np.array(close_value_list)).astype(np.float))
    plt.gcf().autofmt_xdate()
    return fig

def tweet_graph(polarity_list):
    count_list = []
    for i in range(len(polarity_list)):
        count_list.append(i)
    plt.clf()
    fig = Figure()
    fig.set_size_inches(10.5, 7.5)
    fig.clear()
    axes = fig.add_subplot(1,1,1)
    plt.xlabel('Tweet', fontsize=18)
    plt.ylabel('Polarity', fontsize=16)
    axes.plot(count_list, polarity_list)
    return fig

def min_max_values(daily_stock_all, company_name):
    close_value_list = []
    count_list = []
    days = []
    count = 100
    for key, value in daily_stock_all.items():
        closing_price = float(value['4. close'])
        close_value_list.append(closing_price)
        count_list.append(count)
        days.append(key)
        count = count - 1
    min_stock = min(close_value_list)
    max_stock = max(close_value_list)
    average_stock = statistics.mean(close_value_list) 
    return [min_stock, max_stock, average_stock]

def retrieve_news_info(all_news):
    all_headline_titles = []
    all_headline_links = []
    all_headline_image_links = []
    all_headline_authors = []
    all_headline_descriptions = []
    all_headline_publish_dates = []
    for article in all_news:
        all_headline_titles.append(article['title'])
        all_headline_links.append(article['url'])
        all_headline_image_links.append(article['urlToImage'])
        all_headline_authors.append(article['author'])
        all_headline_descriptions.append(article['description'])
        all_headline_publish_dates.append(article['publishedAt'])
    return [all_headline_titles, all_headline_links, all_headline_image_links, all_headline_authors, all_headline_descriptions,all_headline_publish_dates]

def detect_news_sentiment(all_news):
    news_polarity_list = []
    all_headline_content = []
    all_headline_titles = []
    total_news_polarity = 0
    for article in all_news:
        title = article['title']
        content = article['content']
        all_headline_titles.append(article['title'])
        all_headline_content.append(article['content'])
        for sentence in TextBlob(title).sentences:
            news_polarity_list.append(sentence.sentiment.polarity)

        for sentence in TextBlob(content).sentences:
            news_polarity_list.append(sentence.sentiment.polarity)
    total_news_polarity = statistics.mean(news_polarity_list)
    return [news_polarity_list, total_news_polarity]

            
def detect_tweet_sentiment(ticker):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api_call = tweepy.API(auth)
    polarity_list = []
    tweet_list = []
    tweet_user_list = []
    tweet_user_pic_list = []
    # access_tweets = api_call.search(q=str(ticker), tweet_mode='extended', lang='en', count=100)
    access_tweets = []
    total_polarity = 0
    for tweet in access_tweets:
        tweet_list.append(tweet.full_text)
        tweet_user_list.append(str(tweet.user._json['name']))
        tweet_user_pic_list.append(str(tweet.user._json['profile_image_url']))
        for sentence in TextBlob(tweet.full_text).sentences:
            polarity_list.append(sentence.sentiment.polarity)
    # total_polarity = statistics.mean(polarity_list)
    total_polarity = 0
    return [total_polarity, polarity_list, tweet_list, tweet_user_list, tweet_user_pic_list]

def final_tweet_recommendation(total_polarity, company_name):
    if total_polarity > 0:
        return ("After analyzing recent tweets about %s , it would be a good idea to invest now!" % company_name)
    elif total_polarity < 0:
        return ("After analyzing recent tweets about %s , it would be a bad idea to invest now!" % company_name)
    else:
        return ("Looks like Twitter can't decide how they felt about %s . Sorry!" % company_name)

def final_news_recommendation(total_news_polarity, company_name):
    if total_news_polarity > 0:
        return ("After analyzing recent news about %s , it would be a good idea to invest now!" % company_name)
    elif total_news_polarity < 0:
        return ("After analyzing recent news about %s , it would be a bad idea to invest now!" % company_name)
    else:
        return ("Looks like the last media cycle can't decide how they felt about %s . Sorry!" % company_name)
    
def retrieve_competitor_bar_chart(tickerList):
    stock_prices = []

    for ticker in tickerList:
        stock_JSON = requests.get(url + ticker + url_2).json()
        daily_stock_all = stock_JSON['Time Series (Daily)']
    
        price = []
        for key, value in daily_stock_all.items():
            price.append(value['4. close'])
    
        stock_prices.append((np.array(price)).astype(np.float))
    min_arr = []
    max_arr = []
    avg_arr = []
    for stock_price in stock_prices:
        min_arr.append(np.min(stock_price))
        max_arr.append(np.max(stock_price))
        avg_arr.append(np.average(stock_price))
    
    df = pandas.DataFrame(dict(graph=tickerList, n=min_arr, a=avg_arr, m=max_arr)) 

    ind = np.arange(len(df))
    width = 0.3

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.barh(ind, df.n, width, color='red', label='Minimum Value')
    ax.barh(ind + width, df.a, width, color='green', label='Average Value')
    ax.barh(ind + 2*width, df.m, width, color='blue', label='Maximum Value')

    ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[3*width - 1, len(df)])
    ax.legend()
    return ax

if __name__ == '__main__':
    app.run(debug=True)