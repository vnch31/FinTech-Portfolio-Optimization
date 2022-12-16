import numpy as np
import pandas as pd
import yfinance
import os
import logging
import sys
import matplotlib.pyplot as plt

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
    print(arr_symbol)

    # arr_symbol = ['TSLA', 'BIO', 'TGT']
    # arr_symbol = ['MMM', 'AOS', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADM', 'ADBE', 'AAP', 'AMD', 'AES']
    arr_symbol = ['MMM', 'AOS', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADM', 'ADBE', 'AAP', 'AMD', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'CHRW', 'CDNS', 'CZR', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FRC', 'FE', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GS', 'HAL', 'HBI', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PBCT', 'PEP', 'PKI', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'QCOM', 'PWR', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'COO', 'HIG', 'HSY', 'MOS', 'TRV', 'DIS', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UAA', 'UA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']
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

        # save to file
        print(f'./data/autotickers/{filename}')
        with open(f'./data/autotickers/{filename}', 'w+') as fd:
            df.to_csv(fd)

        logging.debug(f"Dataset saved as : {filename}")

    total = pd.DataFrame()

    for ticker in arr_symbol:
      # ticker = yfinance.Ticker(symbol)
      # close = ticker.history(interval="1d",start=start_date,end=end_date)[['Close']].pct_change(1)
      

        # return and log return
        # df.loc[df['Ticker'] == ticker, 'Return'] = df.loc[df['Ticker'] == ticker, 'Close'].pct_change(1)

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
    # print(df)


    f = plt.figure(figsize=(15,10))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix top ' + str(num_tickers * 2), fontsize=16,pad=30);
    plt.savefig('top_'+ str(num_tickers * 2) + '_' + filename_plt)

    return list_top_num

def cache_news(top_n_tickers_from_s500):
    column = ["title"]
    subreddits = ['StocksAndTrading', 'Wallstreetbetsnew', 'wallstreetbetsOGs', 'daytrading', 'StockMarket', 'stocks', 'dividends']

    for tick in top_n_tickers_from_s500:
      if len(tick) < 2:
        continue
      file_news = f'./data/autotickers/{tick}.csv'
      if os.path.isfile(file_news):
        print("Already cached: " + file_news)
        continue
      print(f"extracting and analyzing data for slock: {tick}")
      df_news = pd.DataFrame(columns=['ticker', 'title'])
      for sub_reddit in subreddits:
        api = "https://api.pushshift.io/reddit/search/submission/?q="+tick+"&subreddit="+sub_reddit+"&size=500&fields="+",".join(column)
        res = requests.get(api, timeout=10)
        json_data = res.json()['data']
        for itr in range(len(json_data)):
          entry = json_data[itr]
          text = entry['title']
          df_news = df_news.append({'ticker': tick, 'title': text}, ignore_index=True)
      df_news.to_csv(file_news)
      print(df_news)

def get_sentiment(start_date, end_date, top_n_tickers_from_s500, num_tickers):
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # column = ["id","subreddit", "subreddit_id", "subreddit_subscribers","author", "created_utc", "full_link", "score", "title", "upvote_ratio", "url"]
    column = ["title"]

    subreddits = ['StocksAndTrading', 'Wallstreetbetsnew', 'wallstreetbetsOGs', 'daytrading', 'StockMarket', 'stocks', 'dividends']
    df_results = pd.DataFrame(columns=['stock', 'positive', 'negative'])

    for tick in top_n_tickers_from_s500:
      if len(tick) < 2:
        continue
      print(f"extracting and analyzing data for slock: {tick}")
      tick_pos = 0
      tick_neg = 0

      file_news = f'./data/autotickers/{tick}.csv'
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
      print(f"tick pos: {tick_pos}, tick neg: {tick_neg}")
      df_results = df_results.append(df_res, ignore_index=True)
    df_results['sentiment'] = df_results['positive'] - df_results['negative']
    df_results = df_results.sort_values(by=['sentiment'], ascending=False).reset_index(drop=True).head(num_tickers)
    print(df_results)
    sorted_list_of_stocks = df_results.sort_values(by=['stock'], ascending=True)
    return sorted_list_of_stocks['stock'].values.tolist()

def get_auto_tickers(start_date, end_date, num_tickers =10, interval="1d"):
  list_candidate_tickers_from_s500 = get_data_from_sp500(start_date, end_date, num_tickers)
  print(list_candidate_tickers_from_s500)
  cache_news(list_candidate_tickers_from_s500)
  top_n_tickers_from_s500_with_sentiment = get_sentiment(start_date, end_date, list_candidate_tickers_from_s500, num_tickers)

  print("Tickers with maximum variance & positive sentiment")
  print(top_n_tickers_from_s500_with_sentiment)
  return top_n_tickers_from_s500_with_sentiment
