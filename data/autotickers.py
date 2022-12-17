import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import yfinance
import os
import logging
import sys
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews

# for sentiment analysis
from transformers import pipeline
import requests


# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def get_data_from_sp500(start_date, end_date, num_tickers, interval="1d"):


    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    tickers = pd.read_csv(url)
    ticker = yfinance.Ticker("SPY")
    days = ticker.history(interval="1d",start=start_date,end=end_date).shape[0]


    arr_symbol = []
    for symbol in tickers['Symbol']:
      arr_symbol.append(symbol)

    logging.debug(f"Tickers to evaluate: {arr_symbol}")

    filename = f"cache_auto_tickers_{start_date}_{end_date}.csv"
    filename_plt = f"auto_tickers_plt_{start_date}_{end_date}.png"
    df = None

    # check if file exists
    if os.path.isfile(f'./data/autotickers/{filename}'):
        logging.debug(f'File {filename}, no need to download, loading file...')
        df = pd.read_csv(f'./data/autotickers/{filename}')
        df = df.sort_values(by=['Date', 'Ticker'])
        logging.debug(df)

    else:
        # downloading data 
        df = yfinance.download(
            tickers=arr_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=True
        )
        # remove multi level column
        df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        df = df.sort_values(by=['Date', 'Ticker'])

        logging.debug(df)

        # save to file (cache)
        logging.debug(f'./data/autotickers/{filename}')

        with open(f'./data/autotickers/{filename}', 'w+') as fd:
            df.to_csv(fd)

        logging.debug(f"Dataset saved as : {filename}")

    total = pd.DataFrame()

    for ticker in arr_symbol:

        tmp = df.loc[df['Ticker'] == ticker, 'Close'].pct_change(1)
        tmp = tmp.dropna().reset_index(drop=True)
        tmp_pd = pd.DataFrame(tmp)
        tmp_pd.set_axis([ticker], axis="columns", inplace=True)
        # tmp = tmp.drop(['Ticker', 'Close','High','Low','Open','Volume'], axis=1)
        total = pd.merge(total, tmp_pd,left_index = True, right_index = True, how='outer')

    corr_table = total.corr()
    corr_table['stock1'] = corr_table.index
    corr_table = corr_table.melt(id_vars = 'stock1', var_name = "stock2").reset_index(drop = True)

    corr_table = corr_table[corr_table['stock1'] < corr_table['stock2']].dropna()
    corr_table['abs_value'] = np.abs(corr_table['value'])

    highest_corr = corr_table.sort_values("abs_value",ascending = True).head(num_tickers)
    top_num_uncorr = pd.DataFrame(highest_corr['stock1'].append(highest_corr['stock2'],ignore_index = True),columns=['stock']).drop_duplicates()

    list_top_num = list(top_num_uncorr.values.flatten())

    df = total[list_top_num] 

    f = plt.figure(figsize=(15,10))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix top ' + str(num_tickers * 2), fontsize=16,pad=30);
    save_path = './data/autotickers/'+ 'correlation_matrix_'+ str(num_tickers * 2) + '_' + filename_plt
    plt.savefig(save_path)

    return list_top_num

def cache_news(start_date, end_date, top_n_tickers_from_s500):
    column = ["title"]

    for tick in top_n_tickers_from_s500:
      if len(tick) < 2:
        continue
      file_news = f'./data/autotickers/news-{tick}.csv'
      if os.path.isfile(file_news):
        logging.debug(f"Already cached: {file_news}")

        continue
      logging.debug(f"extracting and analyzing data for stock: {tick}")
      df_news = pd.DataFrame(columns=['ticker', 'title'])

      # source: REDDIT
      # Lists of Sub-Reddit
      subreddits = ['StocksAndTrading', 'Wallstreetbetsnew', 'wallstreetbetsOGs', 'daytrading', 'StockMarket', 'stocks', 'dividends']
      for sub_reddit in subreddits:
        api = "http://api.pushshift.io/reddit/search/submission/?q="+tick+"&subreddit="+sub_reddit+"&size=500&fields="+",".join(column)
        try:
          res = requests.get(api, timeout=3)
          json_data = res.json()
          print(json_data)
          if json_data['data'] and len(json_data['data']) > 0:
            for itr in range(len(json_data['data'])):
              entry = json_data['data'][itr]
              text = entry['title']
              df_news = df_news.append({'ticker': tick, 'title': text}, ignore_index=True)
        except requests.exceptions.RequestException as e:
          logging.debug(f"Skip search Reddit on: {tick} subreddit: {sub_reddit}")

      # source: Google News 
      # Lists of Media Source

      media_lists = ['Yahoo Finance', 'CNBC', 'Bloomberg.com', 'Seeking Alpha', 'Nasdaq']

      googlenews = GoogleNews()
      googlenews.set_lang('en')
      googlenews.set_encode('utf-8')
      googlenews.set_time_range(start_date, end_date)

      keyword = tick
      googlenews.search(keyword)
      results = googlenews.results()

      arr_title = []
      for item in results:
        if item['media'] in media_lists:
          df_news = df_news.append({'ticker': tick, 'title': item['title']}, ignore_index=True)

      df_news.to_csv(file_news)
      
def get_sentiment(start_date, end_date, top_n_tickers_from_s500, num_tickers):
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    column = ["title"]
    df_results = pd.DataFrame(columns=['stock', 'positive', 'negative'])

    for tick in top_n_tickers_from_s500:
      if len(tick) < 2:
        continue

      logging.debug(f"extracting and analyzing data for stock: {tick}")

      tick_pos = 0
      tick_neg = 0

      file_news = f'./data/autotickers/news-{tick}.csv'
      df_titles = pd.read_csv(file_news)
      
      for index, row in df_titles.iterrows():
        title = row['title']
        SA_res = classifier(title)
        print(title)
        print(SA_res)
        if SA_res[0]['label'] == 'POSITIVE':
          tick_pos += 1
        elif SA_res[0]['label'] == 'NEGATIVE':
          tick_neg += 1

      df_res = pd.DataFrame(data=[[tick,tick_pos,tick_neg]], columns=['stock', 'positive', 'negative'])
      logging.debug(f"tick pos: {tick_pos}, tick neg: {tick_neg}")
      df_results = df_results.append(df_res, ignore_index=True)

    df_results['sentiment'] = df_results['positive'] - df_results['negative']
    logging.debug(df_results)
    df_results = df_results.sort_values(by=['sentiment'], ascending=False).reset_index(drop=True).head(num_tickers)
    logging.debug(f"Ticker Lists:")
    print(df_results)
    
    sorted_list_of_stocks = df_results.sort_values(by=['stock'], ascending=True)
    return sorted_list_of_stocks['stock'].values.tolist()

def get_auto_tickers(start_date, end_date, num_tickers =10, interval="1d"):
  list_candidate_tickers_from_s500 = get_data_from_sp500(start_date, end_date, num_tickers)
  cache_news(start_date, end_date, list_candidate_tickers_from_s500)
  top_n_tickers_from_s500_with_sentiment = get_sentiment(start_date, end_date, list_candidate_tickers_from_s500, num_tickers)

  logging.debug(f"Tickers with maximum variance & positive sentiment: {top_n_tickers_from_s500_with_sentiment}")
  return top_n_tickers_from_s500_with_sentiment
