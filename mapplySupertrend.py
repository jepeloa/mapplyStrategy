import numpy as np
import pandas as pd
import talib.abstract as ta
import logging
from hmmlearn.hmm import GaussianHMM
from freqtrade.strategy import IStrategy

# Configuramos el logger de Freqtrade
logger = logging.getLogger(__name__)

class mapplySupertrend(IStrategy):
    # Parámetros básicos de la estrategia
    timeframe = '5m'
    minimal_roi = {"0": 0.20}
    stoploss = -0.10

    # Parámetros para indicadores técnicos
    rsi_period = 14
    ema_period = 21

    # En modo normal (live), se entrena usando todo el dataframe disponible.
    # Se elimina el filtrado de los primeros X meses.
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Cálculo de indicadores técnicos básicos
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=self.ema_period)
        dataframe['ema_diff'] = dataframe['close'] - dataframe['ema21']
        dataframe['returns'] = dataframe['close'].pct_change()

        # Se utiliza el dataframe completo para el entrenamiento
        df_train = dataframe.copy()
        logger.info(f"Using entire dataframe for training (total {len(df_train)} rows)")

        # Seleccionamos las filas con datos válidos para el entrenamiento
        features_train = df_train[['returns', 'rsi', 'ema_diff']].dropna()
        if len(features_train) < 50:
            logger.info("Not enough training data (need at least 50 rows)")
            dataframe['hmm_state'] = np.nan
            dataframe['hmm_trend'] = np.nan
            return dataframe

        X_train = features_train.values

        # Entrenamos el modelo HMM con 3 estados
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=5000, random_state=42)
        model.fit(X_train)
        logger.info("HMM model trained on training data using entire dataset")

        # Definición interna del mapeo de estados según los retornos medios (sin loguear las etiquetas)
        states_train = model.predict(X_train)
        state_returns = {}
        for state in range(3):
            if np.any(states_train == state):
                state_returns[state] = np.mean(X_train[states_train == state, 0])
            else:
                state_returns[state] = float('inf')
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        state_mapping = {
            sorted_states[0][0]: "bajista",
            sorted_states[1][0]: "lateral",
            sorted_states[2][0]: "alcista"
        }

        # Predecimos los estados para todo el dataframe usando el modelo entrenado
        features_all = dataframe[['returns', 'rsi', 'ema_diff']].dropna()
        X_all = features_all.values
        hmm_states = model.predict(X_all)

        # Asignamos los resultados al dataframe
        dataframe['hmm_state'] = np.nan
        dataframe['hmm_trend'] = np.nan
        valid_index = features_all.index
        dataframe.loc[valid_index, 'hmm_state'] = hmm_states
        dataframe.loc[valid_index, 'hmm_trend'] = [state_mapping[s] for s in hmm_states]

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Condiciones de entrada:
         - Se compra cuando el mercado está en régimen alcista (según HMM)
           y se cumplen condiciones de precio bajo: close < ema21 y rsi < 50.
        """
        dataframe.loc[
            (
                (dataframe['hmm_trend'] == "alcista") &
                (dataframe['close'] < dataframe['ema21']) &
                (dataframe['rsi'] < 50)
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Condiciones de salida:
         - Se vende en régimen bajista si el precio es alto (close > ema21 y rsi > 50)
         - Se vende en régimen lateral si el RSI indica sobrecompra (rsi > 70)
        """
        dataframe.loc[
            (
                (dataframe['hmm_trend'] == "bajista") &
                (dataframe['close'] > dataframe['ema21']) &
                (dataframe['rsi'] > 50)
            ),
            'sell'
        ] = 1

        dataframe.loc[
            (
                (dataframe['hmm_trend'] == "lateral") &
                (dataframe['rsi'] > 70)
            ),
            'sell'
        ] = 1

        return dataframe
