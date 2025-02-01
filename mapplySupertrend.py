import logging
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from freqtrade.persistence import Trade
import datetime
logger = logging.getLogger(__name__)

class mapplySupertrend(IStrategy):
    INTERFACE_VERSION: int = 3
    
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    minimal_roi = {
        "0": 0.05,
        "372": 0.03,
        "861": 0.01,
        "2221": 0
    }

    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    timeframe = '1h'
    startup_candle_count = 200

    buy_m1 = IntParameter(1, 7, default=4, space='buy')
    buy_m2 = IntParameter(1, 7, default=4, space='buy')
    buy_m3 = IntParameter(1, 7, default=4, space='buy')
    buy_p1 = IntParameter(7, 21, default=14, space='buy')
    buy_p2 = IntParameter(7, 21, default=14, space='buy')
    buy_p3 = IntParameter(7, 21, default=14, space='buy')

    sell_m1 = IntParameter(1, 7, default=4, space='sell')
    sell_m2 = IntParameter(1, 7, default=4, space='sell')
    sell_m3 = IntParameter(1, 7, default=4, space='sell')
    sell_p1 = IntParameter(7, 21, default=14, space='sell')
    sell_p2 = IntParameter(7, 21, default=14, space='sell')
    sell_p3 = IntParameter(7, 21, default=14, space='sell')


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calcular supertrends para la configuración actual
        supertrend_1_buy = self.supertrend(dataframe, self.buy_m1.value, self.buy_p1.value)
        supertrend_2_buy = self.supertrend(dataframe, self.buy_m2.value, self.buy_p2.value)
        supertrend_3_buy = self.supertrend(dataframe, self.buy_m3.value, self.buy_p3.value)

        supertrend_1_sell = self.supertrend(dataframe, self.sell_m1.value, self.sell_p1.value)
        supertrend_2_sell = self.supertrend(dataframe, self.sell_m2.value, self.sell_p2.value)
        supertrend_3_sell = self.supertrend(dataframe, self.sell_m3.value, self.sell_p3.value)

        dataframe['supertrend_1_buy'] = supertrend_1_buy['STX']
        dataframe['supertrend_2_buy'] = supertrend_2_buy['STX']
        dataframe['supertrend_3_buy'] = supertrend_3_buy['STX']

        dataframe['supertrend_1_sell'] = supertrend_1_sell['STX']
        dataframe['supertrend_2_sell'] = supertrend_2_sell['STX']
        dataframe['supertrend_3_sell'] = supertrend_3_sell['STX']

        # EMA 200 (para filtro de tendencia, ajustado a 20 como en el código original)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=20)

        # ADX + DI+ y DI-
        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['DI_plus'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['DI_minus'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        dataframe['sar'] = ta.SAR(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Filtro ADX: Por ejemplo, ADX > 50 para mayor fortaleza de tendencia
        adx_threshold = 0

        dataframe.loc[
            (
               (dataframe['supertrend_1_sell'] == 'up') &
               (dataframe['supertrend_2_sell'] == 'up') &
               (dataframe['supertrend_3_sell'] == 'up') &
               (dataframe['ADX'] > adx_threshold) &
               #(dataframe['DI_plus'] > dataframe['DI_minus']) &
               (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Actualmente sin condiciones específicas, se puede mejorar.
        #dataframe.loc[
         #   (
               #(dataframe['supertrend_1_sell'] == 'down') &
               #(dataframe['supertrend_2_sell'] == 'down') &
               #(dataframe['supertrend_3_sell'] == 'down') &
               #(dataframe['ADX'] > adx_threshold) &
               #(dataframe['DI_plus'] > dataframe['DI_minus']) &
          #     (dataframe['volume'] > 0)
           # ),
            #'exit_long'] = 1

        return dataframe

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = f'ST_{period}_{multiplier}'
        stx = f'STX_{period}_{multiplier}'

        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        df['final_ub'] = 0.00
        df['final_lb'] = 0.00

        for i in range(period, len(df)):
            df['final_ub'].iat[i] = (df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1]
                                     else df['final_ub'].iat[i - 1])
            df['final_lb'].iat[i] = (df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1]
                                     else df['final_lb'].iat[i - 1])

        df[st] = 0.00
        for i in range(period, len(df)):
            if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i]:
                df[st].iat[i] = df['final_ub'].iat[i]
            elif df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i]:
                df[st].iat[i] = df['final_lb'].iat[i]
            elif df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i]:
                df[st].iat[i] = df['final_lb'].iat[i]
            elif df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i]:
                df[st].iat[i] = df['final_ub'].iat[i]

        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST': df[st],
            'STX': df[stx]
        })
