import pandas as pd
import yfinance as yf
import json
import argparse
import os
import logging
import sys

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

def get_data_yfinance(tickers, start_date, end_date, interval):
        logging.debug(f"Fetch data using the following parameters:")
        logging.debug(f"Tickers: {tickers}")
        logging.debug(f"Start date: {start_date}")
        logging.debug(f"End date: {end_date}")
        logging.debug(f"Interval: {interval}")

        filename = f"dataset_{start_date}_{end_date}_{tickers.replace(' ', '_')}.csv"

        # check if file exists
        if os.path.isfile(f'./data/downloads/{filename}'):
            logging.debug(f'File {filename}, no need to download, loading file...')
            df = pd.read_csv(f'./data/downloads/{filename}', index_col='Date')
            df.index = pd.to_datetime(df.index)
            df.index = pd.DatetimeIndex(df.index)
            logging.debug(df)
            return df

        # downloading data 
        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            ignore_tz = True,
            group_by='ticker',
            auto_adjust=True
        )

        # remove multi level column
        if len(tickers.split(' ')) > 1:
            df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

        logging.debug(df)

        # check if save exists
        if not os.path.exists('./data/downloads/'):
            logging.debug("Creating save directory")
            os.chdir('data')
            os.mkdir('downloads')
            os.chdir('..')

        # save to file
        with open(f'./data/downloads/{filename}', 'w+') as fd:
            df.to_csv(fd)

        logging.debug(f"Dataset saved as : {filename}")

        return df



