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
            df = pd.read_csv(f'./data/downloads/{filename}', index_col=0)
            logging.debug(df)
            return df

        # downloading data 
        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=True
        )
        # remove multi level column
        df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        df = df.reset_index()

        logging.debug(df)

        # save to file
        with open(f'./data/downloads/{filename}', 'w+') as fd:
            df.to_csv(fd)

        logging.debug(f"Dataset saved as : {filename}")

        return df



