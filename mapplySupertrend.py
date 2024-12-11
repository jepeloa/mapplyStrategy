from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, merge
import talib.abstract as ta
from functools import reduce
import pandas as pd

class mapplySupertrend(IStrategy):
    # Configuración de ROI decreciente por tiempo
    minimal_roi = {
        "0": 0.03,
        "30": 0.03,
        "60": 0.03,
        "120": 0.03
    }

    # Configuración de otros parámetros
    timeframe = '5m'
    informative_timeframes = ['15m', '1h', '4h']
    stoploss = -0.20
    max_open_trades = 120

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            for timeframe in self.informative_timeframes:
                informative_pairs.append((pair, timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicadores en el marco de tiempo base (5m)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe['stoch_rsi'] = stoch_rsi['fastk']
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        # Normalizar indicadores al rango 0-1
        dataframe['rsi_norm'] = dataframe['rsi'] / 100
        dataframe['mfi_norm'] = dataframe['mfi'] / 100
        dataframe['stoch_rsi_norm'] = dataframe['stoch_rsi'] / 100

        # Obtener indicadores en marcos de tiempo informativos
        for timeframe in self.informative_timeframes:
            informative_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=timeframe)

            # Calcular indicadores en el marco de tiempo informativo
            informative_df[f'rsi_{timeframe}'] = ta.RSI(informative_df, timeperiod=14)
            informative_df[f'mfi_{timeframe}'] = ta.MFI(informative_df, timeperiod=14)
            stoch_rsi_inf = ta.STOCHRSI(informative_df, timeperiod=14)
            informative_df[f'stoch_rsi_{timeframe}'] = stoch_rsi_inf['fastk']

            # Normalizar indicadores al rango 0-1
            informative_df[f'composite_signal_{timeframe}'] = (
                informative_df[f'rsi_{timeframe}'] / 100 * 
                informative_df[f'mfi_{timeframe}'] / 100 * 
                informative_df[f'stoch_rsi_{timeframe}'] / 100
            )

            # Seleccionar columnas necesarias
            informative_df = informative_df[['date', f'composite_signal_{timeframe}']]

            # Fusionar manualmente el DataFrame informativo con el DataFrame principal
            dataframe = merge(dataframe, informative_df, on='date', how='left')

        # Señal compuesta por multiplicación de indicadores en el marco de tiempo base
        dataframe['composite_signal'] = dataframe['rsi_norm'] * dataframe['mfi_norm'] * dataframe['stoch_rsi_norm']

        print(dataframe.columns)  # Para verificar las columnas resultantes
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Umbral para la señal compuesta
        composite_threshold = 0

        # Condiciones de compra
        conditions = [
            dataframe['composite_signal'] > composite_threshold
        ]

        # Condiciones adicionales con marcos de tiempo informativos
        if 'composite_signal_4h' in dataframe.columns and 'composite_signal_1h' in dataframe.columns:
            condition_1 = dataframe['composite_signal_4h'] > dataframe['composite_signal_1h']
            conditions.append(condition_1)
            print("Condition 1 (4h > 1h):", condition_1.sum())

        if 'composite_signal_15m' in dataframe.columns and 'composite_signal_1h' in dataframe.columns:
            condition_2 = dataframe['composite_signal_15m'] > dataframe['composite_signal_1h']
            conditions.append(condition_2)
            print("Condition 2 (15m > 1h):", condition_2.sum())

        if 'composite_signal' in dataframe.columns and 'composite_signal_15m' in dataframe.columns:
            condition_3 = dataframe['composite_signal'] > dataframe['composite_signal_15m']
            conditions.append(condition_3)
            print("Condition 3 (composite > 15m):", condition_3.sum())

        # Aplicar condiciones para la señal de compra si se cumplen todas
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Placeholder para lógica de venta, si es necesario implementarla
        #  composite_threshold = 0.8

        # # Condiciones de compra
        #  conditions = [
        #     dataframe['composite_signal'] > composite_threshold
        #  ]
        #  if conditions:
        #      dataframe.loc[
        #          reduce(lambda x, y: x & y, conditions),
        #          'sell'] = 1
             
         return dataframe
