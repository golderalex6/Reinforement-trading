import pandas as pd

import os
from pathlib import Path
import datetime as dt

from tvDatafeed import TvDatafeed,Interval

class StockData():
    def __init__(self) -> None:
        """
        Initializes the instance with a TV data feed and a dictionary for timeframe conversions.

        This method sets up the TV data feed using the TvDatafeed class and initializes a mapping
        between time intervals (in string format) and corresponding values from the Interval class.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """

        self._tv=TvDatafeed()
        self._timeframe_convert={
                '1min':Interval.in_1_minute,'3min':Interval.in_3_minute,'5min':Interval.in_5_minute,
                '15min':Interval.in_15_minute,'30min':Interval.in_30_minute,'45min':Interval.in_45_minute,
                '1h':Interval.in_1_hour,'2h':Interval.in_2_hour,'3h':Interval.in_3_hour,'4h':Interval.in_4_hour,
                '1d':Interval.in_daily
            }

    def _clean_data(self,df:pd.DataFrame,timeframe:str) -> pd.DataFrame:
        """
        Cleans and processes the input DataFrame by resetting the index, renaming columns, and adding necessary fields.

        This method formats the given DataFrame by resetting its index, dropping unnecessary columns, renaming columns,
        adding a 'Timestamp' column, and including a 'Timeframe' column. The DataFrame is then reordered and set to use
        'Datetime' as the index.

        Parameters:
        -----------
            df (pd.DataFrame): The input DataFrame containing raw financial data.
            timeframe (str): The timeframe of the data, used to add a 'Timeframe' column.

        Returns:
        --------
            pd.DataFrame: The cleaned and processed DataFrame with additional columns and proper formatting.
        """

        df=df.reset_index()
        df=df.drop(columns=['symbol'])
        df.columns=['Datetime','Open','High','Low','Close','Volume']
        df['Timestamp']=df['Datetime'].map(lambda x:x.timestamp())
        df['Timeframe']=timeframe
        df=df[['Datetime','Timestamp','Timeframe','Open','High','Low','Close','Volume']]
        df=df.set_index('Datetime')
        return df

    def get_data(self,companies:dict) -> None:
        """
        Retrieves historical data for a list of companies and saves it to CSV files.

        This method iterates over the provided dictionary of companies, where each company is represented by a symbol
        and exchange. It checks if a directory for each company exists, creates it if necessary, and then fetches historical
        data for each timeframe. The data is cleaned and saved as CSV files in the corresponding company directory.

        Parameters:
        -----------
            companies (dict): A dictionary where keys are company symbols and values are exchange names.

        Returns:
        --------
            None
        """

        for symbol,exchange in companies.items():
            if not os.path.exists(os.path.join(Path(__file__).parent,f'data/{symbol}')):
                os.mkdir(os.path.join(Path(__file__).parent,f'data/{symbol}'))
            for timeframe,interval_object in self._timeframe_convert.items():
                print(symbol,timeframe)
                try:
                    df=self._tv.get_hist(symbol,exchange=exchange,interval=interval_object,n_bars=3000)
                    df=self._clean_data(df,timeframe)
                    df.to_csv(os.path.join(Path(__file__).parent,f'data/{symbol}/{symbol}_{timeframe}.csv'))
                except Exception as e:
                    print(f'Error on {symbol} {str(e)}')

if __name__=='__main__':
    companies={
            'TSLA':'NASDAQ',
            'MSFT':'NASDAQ',
            'AAPL':'NASDAQ',
        }
    stock=StockData()
    stock.get_data(companies)

