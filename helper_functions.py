from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection._search import _check_param_grid, check_cv, check_scoring, is_classifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import timedelta
import datetime, calendar, workdays
import sys, os, re
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, USMartinLutherKingJr, \
     USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
try:
    import blpwrapper
    import pdblp
except:
    pass

#------------------------------------------------------------------------
# Pipeline - Feature Buildout
#------------------------------------------------------------------------

EQUITY_TICKERS   = ['spy', 'iwm', 'mdy', 'hyg'] #'xiv', 'vxx', 'tvix', 
INDUSTRY_TICKERS = ['xle', 'xlb', 'xlf', 'iyz', 'xlv', 'xlk', 'xlp', 'xlu', 'xly', 'xli']
INDEX_TICKERS    = ['spvixstr', 'sx5e', 'vix', 'pcrteqty', 'vvix', 'vxv', 'vxth', 'rxm', 'rvx', 'vxn',
                    'tyvix', 'jpmvxyg7', 'jpmvxyem', 'tradhigh', 'tradlows', 'cvxftncn', 'vxxiv', 'cesiusd', 'cesig10']
                    # 'vcac', 'v2x', 'vhsi', 'vaex', 'vimex', 'tradcads', 'vnky', 'vftse', 'vkospi'
MACRO_TICKERS    = ['nfp tch', 'injcjc', 'usmmmnch', 'injcsp', 'ip chng', 'consexp', 'dgnoxtch', 'pitlchng', 'lei chng', 'napmpmi', 'napmnmi', 'conssent', 'concconf']
VIX_FUTURES      = ['ux1', 'ux2', 'ux3', 'ux4', 'ux5', 'ux6', 'ux7']
VOL_FIELDS       = ['30DAY_IMPVOL_97.5%MNY_DF', '30DAY_IMPVOL_102.5%MNY_DF', '30DAY_IMPVOL_95.0%MNY_DF', '30DAY_IMPVOL_105.0%MNY_DF','3MTH_IMPVOL_95.0%MNY_DF', '3MTH_IMPVOL_105.0%MNY_DF']
OTHER            = ['weight']
equity_tks       = [tk + ' equity' for tk in EQUITY_TICKERS]
industry_tks     = [tk + ' equity' for tk in INDUSTRY_TICKERS]
index_tks        = [tk + ' index' for tk in INDEX_TICKERS]
macro_tks        = [tk + ' index' for tk in MACRO_TICKERS]

def refresh_pickle():
    try: cur_dir = os.path.dirname(__file__)
    except: cur_dir = ''
    data = pd.read_pickle(os.path.join(cur_dir, 'historical_data1.pickle'))
    last_date = data.index[-5].strftime('%Y%m%d')

    INDICATORS_WITH_LAG = ['tradhigh index']
    EQUITY_TICKERS   = ['xiv', 'vxx', 'tvix', 'spy', 'iwm', 'mdy', 'hyg']
    INDUSTRY_TICKERS = ['xle', 'xlb', 'xlf', 'iyz', 'xlv', 'xlk', 'xlp', 'xlu', 'xly', 'xli'] 
    INDEX_TICKERS    = ['spvixstr', 'sx5e', 'vix', 'pcrteqty', 'vvix', 'vxv', 'vxth', 'rxm', 'rvx', 'vxn',
                        'tyvix', 'jpmvxyg7', 'jpmvxyem', 'tradhigh', 'tradlows', 'cvxftncn', 'vxxiv', 'cesiusd', 'cesig10', 
                        'vcac', 'v2x', 'vhsi', 'vaex', 'vimex', 'tradcads', 'vnky', 'vftse', 'vkospi']
    MACRO_TICKERS    = ['nfp tch', 'injcjc', 'usmmmnch', 'injcsp', 'ip chng', 'consexp', 'dgnoxtch', 'pitlchng', 'lei chng', 'napmpmi', 'napmnmi', 'conssent', 'concconf']
    VIX_FUTURES      = ['ux1', 'ux2', 'ux3', 'ux4', 'ux5', 'ux6', 'ux7']
    VOL_FIELDS       = ['30DAY_IMPVOL_97.5%MNY_DF', '30DAY_IMPVOL_102.5%MNY_DF', '30DAY_IMPVOL_95.0%MNY_DF', '30DAY_IMPVOL_105.0%MNY_DF','3MTH_IMPVOL_95.0%MNY_DF', '3MTH_IMPVOL_105.0%MNY_DF']
    OTHER            = ['weight']
    equity_tks       = [tk + ' equity' for tk in EQUITY_TICKERS]
    industry_tks     = [tk + ' equity' for tk in INDUSTRY_TICKERS]
    index_tks        = [tk + ' index' for tk in INDEX_TICKERS]
    macro_tks        = [tk + ' index' for tk in MACRO_TICKERS]

    con = pdblp.BCon(debug=False)
    con.start()
    macro_data = con.bdh(macro_tks, 'ACTUAL_RELEASE', start_date=last_date, end_date=datetime.datetime.today().strftime('%Y%m%d'), ovrds=[('RELEASE_DATE_OVERRIDE','1')])
    index_data = con.bdh(index_tks, 'px_last', start_date=last_date, end_date=datetime.datetime.today().strftime('%Y%m%d'))
    equity_data = con.bdh(equity_tks, 'px_last', start_date=last_date, end_date=datetime.datetime.today().strftime('%Y%m%d'))
    industry_data = con.bdh(industry_tks, 'px_last', start_date=last_date, end_date=datetime.datetime.today().strftime('%Y%m%d'))
    vol_data = con.bdh('spy equity', VOL_FIELDS, start_date=last_date, end_date=datetime.datetime.today().strftime('%Y%m%d'))
    if (datetime.datetime.now().time() < datetime.time(9,30)) and (datetime.datetime.now().weekday() < 5):
        try:
            equity_data_pre_open = con.bdh(equity_tks, 'PX_LAST_ALL_SESSIONS', start_date=datetime.datetime.today().strftime('%Y%m%d'), end_date=datetime.datetime.today().strftime('%Y%m%d'))
            industry_data_pre_open = con.bdh(industry_tks, 'PX_LAST_ALL_SESSIONS', start_date=datetime.datetime.today().strftime('%Y%m%d'), end_date=datetime.datetime.today().strftime('%Y%m%d'))
            equity_data_pre_open = equity_data_pre_open.xs('PX_LAST_ALL_SESSIONS', axis=1, level=1)
            industry_data_pre_open = industry_data_pre_open.xs('PX_LAST_ALL_SESSIONS', axis=1, level=1)
        except e as Exception:
            equity_data_pre_open = None
            industry_data_pre_open = None
            pass
    else:
        equity_data_pre_open = None
        industry_data_pre_open = None

    con.stop()
    macro_df = macro_data.xs('ACTUAL_RELEASE', axis=1, level=1)
    index_df = index_data.xs('px_last', axis=1, level=1)
    equity_df = equity_data.xs('px_last', axis=1, level=1)
    industry_df = industry_data.xs('px_last', axis=1, level=1)
    vol_df = vol_data.xs('spy equity', axis=1, level=0)
    vol_df.columns = ['SPY_' + col for col in vol_df.columns]
    new_vix_futures_data = get_vix_futures_prices(start_date=last_date)
    if equity_data_pre_open is not None:
        equity_df = equity_df.append(equity_data_pre_open)
    if industry_data_pre_open is not None:
        industry_df = industry_df.append(industry_data_pre_open)
    new_data = equity_df.join(industry_df, how='outer').join(index_df, how='outer').join(macro_df, how='outer').join(vol_df, how='outer').join(new_vix_futures_data, how='outer')
    new_data.index = pd.to_datetime(new_data.index)
    new_data_new = new_data[~new_data.index.isin(data.index)].dropna(how='all')
    new_data_refresh = new_data[new_data.index.isin(data.index)].dropna(how='all')
    new_data_new['weight'] = new_data_new.index.map(get_contract_weight)
    new_data_new['contract_days_length'] = new_data_new.index.map(lambda x: get_contract_days_length(x, how='total'))
    new_data_new['contract_days_left'] = new_data_new.index.map(lambda x: get_contract_days_length(x, how='remaining'))
    data = data.append(new_data_new)
    data.update(new_data_refresh, join='left')
    data.sort_index(inplace=True)
    data.to_pickle(os.path.join(cur_dir, 'historical_data1.pickle'))

class ShiftColumns(TransformerMixin, BaseEstimator):

    def __init__(self, feature, shift_by):
        self.shift_by = shift_by
        self.feature = feature

    def transform(self, data):
        data[self.feature] = data[self.feature].shift(self.shift_by)
        return data

    def fit(self, *_):
        return self

class FillNA(TransformerMixin, BaseEstimator):

    def __init__(self, value=None, method=None, drop_non_trading_days=False):
        self.value = value
        self.method = method
        self.drop_non_trading_days = drop_non_trading_days
    
    def transform(self, data):
        if self.value is not None:
            return data.fillna(self.value)
        else:
            trading_days = ~data['spy equity'].isnull()
            data = data.fillna(method=self.method)
            if self.drop_non_trading_days:
                data = data[trading_days]
            return data

    def fit(self, *_):
        return self

class FillInf(TransformerMixin, BaseEstimator):

    def __init__(self, value=None):
        self.value = value
    
    def transform(self, data):
        if self.value is not None:
            data = data.replace(np.inf, self.value)
            data = data.replace(-np.inf, self.value)
            return data

    def fit(self, *_):
        return self

class PercentageChange(TransformerMixin, BaseEstimator):

    def __init__(self, features=None):
        self.features = features
        
    def transform(self, data):
        if self.features is None:
            self.features = data.columns
        pct_data = data[self.features].pct_change()
        pct_data.columns = [column + '_pct' for column in pct_data.columns]
        data = data.join(pct_data)
        return data

    def fit(self, *_):
        return self

class NetChange(TransformerMixin, BaseEstimator):

    def __init__(self, features=None):
        self.features = features
        
    def transform(self, data):
        if self.features is None:
            self.features = data.columns
        chg_data = data[self.features].diff()
        chg_data.columns = [column + '_chg' for column in chg_data.columns]
        data = data.join(chg_data)
        return data

    def fit(self, *_):
        return self

class MACD(TransformerMixin, BaseEstimator):

    def __init__(self, features=None, p1=10, p2=2, signal=2):
        self.features = features
        self.p1 = p1
        self.p2 = p2
        self.signal = signal
        
    def transform(self, data):
        if self.features is None:
            self.features = data.columns

        for feature in self.features:
            data[feature + '_macd'] = (data[feature].ewm(span=self.p2).mean() - data[feature].ewm(span=self.p1).mean()).ewm(span=self.signal).mean()

        return data

    def fit(self, *_):
        return self

class EMA(TransformerMixin, BaseEstimator):

    def __init__(self, features=None, ewm=3):
        self.features = features
        self.ewm=ewm
        
    def transform(self, data):
        if self.features is None:
            self.features = data.columns

        for feature in self.features:
            data[feature + '_ema' + str(self.ewm) + 'd'] = data[feature].ewm(span=self.ewm).mean()

        return data

    def fit(self, *_):
        return self

class DayOfYearTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, features=None, ewm=3):
        pass
    
    def transform(self, data):
        return [date.timetuple().tm_yday for date in data.index]

    def fit(self, *_):
        return self

class DateTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, how=None):
        self.how = how
    
    def transform(self, data):
        if self.how == 'day_of_week':
            return np.array([date.isoweekday() for date in data.index]).reshape(-1,1)
        elif self.how == 'day_of_year':
            return np.array([date.timetuple().tm_yday for date in data.index]).reshape(-1,1)
        elif self.how == 'day_of_month':
            return np.array([date.day for date in data.index]).reshape(-1,1)
        elif self.how == 'month':
            return np.array([date.month for date in data.index]).reshape(-1,1)
        else:
            raise ValueError(self.how, 'does not exist.')
            
    def fit(self, *_):
        return self

class FeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, features=None):
        self.features = features

    def transform(self, data):
        if self.features is None:
            self.features = data.columns
        self.mask = [True if col in self.features else False for col in data.columns]
        return data[self.features]

    def fit(self, *_):
        return self

    def get_support(self):
        return self.mask

class DenseTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass
        
    def transform(self, x):
        
        return x.toarray()

    def fit(self, *_):
        return self

class FillNAColumn(TransformerMixin, BaseEstimator):

    def __init__(self, column_to_fill, column_to_copy):
        self.column_to_fill = column_to_fill
        self.column_to_copy = column_to_copy

    def transform(self, data):
        data[self.column_to_fill] = data[self.column_to_fill].fillna(data[self.column_to_copy])
        return data

    def fit(self, *_):
        return self

class Drop(TransformerMixin, BaseEstimator):

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, data):
        data = data.drop(self.columns_to_drop, axis=1)
        return data

    def fit(self, *_):
        return self

class AddOtherFeatures(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def transform(self, df):
        df.loc[:,'spy_MACD'] = (df['spy equity'].ewm(span=12).mean() - df['spy equity'].ewm(span=26).mean()).ewm(span=9).mean()
        df.loc[:,'spy_5d_ema'] = df['spy equity_pct'].ewm(span=5).mean()
        df.loc[:,'spy_5d_rv'] = df['spy equity_pct'].rolling(5).std()*np.sqrt(260)*100
        df.loc[:,'spy_10d_rv'] = df['spy equity_pct'].rolling(10).std()*np.sqrt(260)*100
        df.loc[:,'spy_20d_rv'] = df['spy equity_pct'].rolling(20).std()*np.sqrt(260)*100
        df.loc[:,'spy_30d_rv'] = df['spy equity_pct'].rolling(30).std()*np.sqrt(260)*100
        df.loc[:,'spy_6m_rv'] = df['spy equity_pct'].rolling(30*6).std()*np.sqrt(260)*100
        df.loc[:,'vix_spy_10rv_prem'] = df['vix index'] - df['spy_10d_rv']
        df.loc[:,'vix_spy_20rv_prem'] = df['vix index'] - df['spy_20d_rv']
        df.loc[:,'vix_spy_6mrv_prem'] = df['vix index'] - df['spy_6m_rv']
        df.loc[:,'spy_rv_momentum'] =  df.spy_10d_rv / df.spy_30d_rv
        df.loc[:,'ctg_momentum'] = (df.ctg_weighted.ewm(span=3).mean() - df.ctg_weighted.ewm(span=21).mean()).ewm(span=2).mean()
        # df['ctg_momentum_chg'] = df.ctg_momentum.diff()
        df.loc[:,'spvixstr_1dl'] = df['spvixstr index_pct'].shift(1)
        df.loc[:,'spvixstr_2dl'] = df['spvixstr index_pct'].shift(2)
        df.loc[:,'spvixstr_3dl'] = df['spvixstr index_pct'].shift(3)
        df.loc[:,'spvixstr_4dl'] = df['spvixstr index_pct'].shift(4)
        df.loc[:,'spvixstr_5dl'] = df['spvixstr index_pct'].shift(5)
        # df['spy_autocorr_10d'] = df['spy equity_pct'].rolling(window=10).apply(lambda x: pd.Series(x).autocorr(lag=1))
        # df['spy_autocorr_20d'] = df['spy equity_pct'].rolling(window=20).apply(lambda x: pd.Series(x).autocorr(lag=1))
        # df['spy_autocorr_30d'] = df['spy equity_pct'].rolling(window=30).apply(lambda x: pd.Series(x).autocorr(lag=1))
        # df['vix_autocorr_10d'] = df['vix index_pct'].rolling(window=10).apply(lambda x: pd.Series(x).autocorr(lag=1))
        # df['spy_kurt_10d'] = df['spy equity_pct'].rolling(window=10).kurt()
        # df['spy_kurt_20d'] = df['spy equity_pct'].rolling(window=20).kurt()
        # df['vix_kurt_10d'] = df['vix index_pct'].rolling(window=10).kurt()
        # df['vix_kurt_20d'] = df['vix index_pct'].rolling(window=20).kurt()
        # df['spy_skew_10d'] = df['spy equity_pct'].rolling(window=10).skew()
        # df['spy_skew_20d'] = df['spy equity_pct'].rolling(window=20).skew()
        # df['vix_skew_10d'] = df['vix index_pct'].rolling(window=10).skew()
        # df['vix_skew_20d'] = df['vix index_pct'].rolling(window=20).skew()
        eq_correl_matrix_ = df[[tks + '_pct' for tks in equity_tks]].rolling(10).corr()
        df.loc[:,'spy_iwm_10d_corr'] = eq_correl_matrix_.unstack(1)[('spy equity_pct', 'iwm equity_pct')]
        df.loc[:,'spy_mdy_10d_corr'] = eq_correl_matrix_.unstack(1)[('spy equity_pct', 'mdy equity_pct')]
        df.loc[:,'spy_hyg_10d_corr'] = eq_correl_matrix_.unstack(1)[('spy equity_pct', 'hyg equity_pct')]
        tks_ = [tk + '_pct' for tk in industry_tks]
        ind_corr_df_ = df[tks_].rolling(10).corr().mean(axis=1).unstack(1)
        ind_corr_df_.columns = [col + '_corr' for col in ind_corr_df_.columns]
        df = df.join(ind_corr_df_)
        df.loc[:,'spvixstr_2d_perf'] = (df['spvixstr index_pct']).rolling(2).apply(lambda x: np.prod(x+1)-1).shift(1)
        df.loc[:,'spvixstr_3d_perf'] = (df['spvixstr index_pct']).rolling(3).apply(lambda x: np.prod(x+1)-1).shift(1)
        df.loc[:,'spvixstr_5d_perf'] = (df['spvixstr index_pct']).rolling(5).apply(lambda x: np.prod(x+1)-1).shift(1)
        df.loc[:,'spy_30d_skew_95_105']     = df['SPY_30DAY_IMPVOL_95.0%MNY_DF'] / df['SPY_30DAY_IMPVOL_105.0%MNY_DF']
        df.loc[:,'spy_30d_skew_97.5_102.5'] = df['SPY_30DAY_IMPVOL_97.5%MNY_DF'] / df['SPY_30DAY_IMPVOL_102.5%MNY_DF']
        df.loc[:,'spy_3m_skew_95_105']      = df['SPY_3MTH_IMPVOL_95.0%MNY_DF'] / df['SPY_3MTH_IMPVOL_105.0%MNY_DF']
        # df['vxx_less_vxxiv'] = df['vxx equity'] / df['vxxiv index']
        return df

    def fit(self, *_):
        return self

class AddVIXFeatures(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def transform(self, df):
        df.loc[:,'ctg_0v1'] = df.ux1/df['vix index'] - 1
        df.loc[:,'ctg_1v2'] = df.ux2/df.ux1 - 1  #np.log(df.vx2_close/df.vx1_close)
        df.loc[:,'ctg_2v3'] = df.ux3/df.ux2 - 1
        df.loc[:,'ctg_curvature'] = (df.ux2/df.ux1 - 1) - (df.ux7/df.ux4 - 1) / 3
        df.loc[:,'ux_weighted'] = df.ux1*df.weight + df.ux2*(1-df.weight)
        df.loc[:,'ux_weighted_2'] = df.ux2*df.weight + df.ux3*(1-df.weight)
        df.loc[:,'ux_weighted_3'] = df.ux3*df.weight + df.ux4*(1-df.weight)
        df.loc[:,'ux_weighted_4'] = df.ux4*df.weight + df.ux5*(1-df.weight)
        df.loc[:,'ux_weighted_5'] = df.ux5*df.weight + df.ux6*(1-df.weight)
        df.loc[:,'ux_weighted_6'] = df.ux6*df.weight + df.ux7*(1-df.weight)
        df.loc[:,'ctg_weighted'] = df.ux_weighted/df['vix index'] - 1
        df.loc[:,'ctg_weighted_1v2'] = df.ctg_1v2*df.weight + df.ctg_2v3*(1-df.weight)
        df.loc[:,'ux_weighted_1d_lag'] = df.ux_weighted.shift(1)
        df.loc[:,'ctg_weighted_1d_lag'] = df.ctg_weighted.shift(1)
        df.loc[:,'vix_1d_lag'] = df['vix index'].shift(1)
        df.loc[:,'ux_wgt_prem'] = df.ux_weighted - df['vix index']
        df.loc[:,'contract_days_times_premium'] = df['ux_wgt_prem'] * df['contract_days_length']
        df.loc[:,'contract_days_remaining_ux2ux1_premium'] = (df['ux2'] - df['ux1']) * df['contract_days_left']
        df.loc[:,'contract_days_remaining_ux1vix_premium'] = (df['ux1'] - df['vix index']) * df['contract_days_left']
        return df

    def fit(self, *_):
        return self

class ModelTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, model, probs=True):
        self.model = model
        self.probs = hasattr(model, 'predict_proba')

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        if self.probs:
            return pd.DataFrame(self.model.predict_proba(X)[:, 1])
        else:
            return pd.DataFrame(self.model.predict(X))
    
    def get_params(self, deep=True):
        return dict(model=self.model, probs=self.probs)


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('July 4th', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

#------------------------------------------------------------------------
# Quick helper functions
#------------------------------------------------------------------------
def to_decimal(x, precision=2):
    try:    formatted_string = '{number:0,.{digits}f}'.format(number=x, digits=precision)
    except: formatted_string = x
    return  formatted_string
def to_percent(x, precision=1):
    try:    formatted_string = '{number:0,.{digits}f}%'.format(number=x*100, digits=precision)
    except: formatted_string = x
    return  formatted_string
def to_number(x):
    divisor = 1 if x.find('%') == -1 else 100
    try:    unformatted_string = float(x.replace('%', '').replace(',', '')) / 100
    except: unformatted_string = 0
    return  unformatted_string    
def sharpe(x, days=1):
    return (np.prod(x+1)**(252/len(x)*days)-1) / (np.std(x)*np.sqrt(252))
def sortino(x, days=1):
    return (np.prod(x+1)**(252/len(x)*days)-1) / (np.std(np.clip(x, a_min=None, a_max=0)) * np.sqrt(252))
def calmar(x, days=1):
    return (np.prod(x+1)**(252/len(x)*days)-1) / (np.std(np.clip(x, a_min=None, a_max=0)) * np.sqrt(252))
def return_scorer(y_true, y_pred):
    def sign(a):
        return [1 if i > 0 else -1 for i in a]
    return np.prod(sign(y_pred) * y_true + 1) - 1

#------------------------------------------------------------------------
# Backtesting
#------------------------------------------------------------------------

def backtest_summary(df_results, sample_periods=[]):
    '''Returns a performance summary table a Pandas data frame containing daily return streams.
    
    Parameters
    ----------
    df_results : Pandas DataFrame
        DataFrame containing daily return streams to be summarized.

    sample_periods : list of dictionaries, shape=[{'from':date, 'to':date}], optional
        List of dictionaries that contains periods to be summarized.
        Format example: [{'from':'2010', 'to':'2012'}, {'from':'2012', 'to':'2015'}]

    days : integer, optional -- DEPRECATED
        Annualization parameter, if returns are not daily. Equal to number of days in return period.

    Returns
    ----------
    Data frame containing the summary table.
    '''

    summary = pd.DataFrame(columns=list(df_results))
    cumulative_performance = df_results.expanding(252).apply(lambda x: np.prod(x + 1) - 1)
    annual_performance = df_results.groupby([lambda x: x.year]).apply(lambda x: np.prod(x+1)-1)

    summary.loc['Annual Returns'] = ''
    summary = summary.append(annual_performance)
    summary.loc[''] = ''

    summary.loc['Max Drawdowns'] = ''
    annual_max_drawdowns = df_results.groupby([lambda x: x.year]).apply(lambda x: np.min(np.cumprod(x + 1) / np.maximum.accumulate(np.cumprod(x + 1))-1))
    annual_max_drawdowns.index = [str(y) + ' MDD' for y in annual_max_drawdowns.index]
    summary = summary.append(annual_max_drawdowns)
    summary.loc['Max Drawdown'] = np.min(np.cumprod(df_results+1) / np.maximum.accumulate(np.cumprod(df_results+1)) - 1)
    summary = summary.applymap(to_percent)
    summary.loc[' '] = ''
    
    for sample_period in sample_periods:
        from_date, to_date = sample_period['from'], sample_period['to']
        temp = pd.DataFrame(columns=list(df_results))
        temp.loc[from_date + ":" + to_date +" Summary",:] = ''
        temp.loc["Total Return",:]  = df_results[from_date:to_date].apply(lambda x: np.prod(x+1)-1).map(lambda x: to_percent(x, precision=0))
        
        ann_factor = 252/df_results.index.freq.n if df_results.index.freq else 252
        
        temp.loc["CAGR",:]          = df_results[from_date:to_date].apply(lambda x: np.prod(x+1)**(ann_factor/len(x))-1).map(to_percent)
        temp.loc['Max Drawdown',:]  = np.min(np.cumprod(df_results[from_date:to_date]+1)
                                      / np.maximum.accumulate(np.cumprod(df_results[from_date:to_date]+1))-1).map(to_percent)
        temp.loc['Sharpe',:]  = df_results[from_date:to_date].apply(sharpe).map(to_decimal)
        temp.loc['Sortino',:] = df_results[from_date:to_date].apply(sortino).map(to_decimal)
        temp.loc['Calmar',:]  = (df_results[from_date:to_date].apply(lambda x: np.prod(x+1)**(ann_factor/len(x))-1) / np.abs(temp.loc['Max Drawdown',:].apply(to_number))).map(to_decimal)
        temp.loc[''] = ''
        summary = summary.append(temp)
    summary = summary.sort_index(axis=1)
    return summary

def plot_returns(df, days_history=252, log_scale=False):
    '''Plots return performance over the speicified time period.
    
    Parameters
    ----------
    df : Pandas DataFrame
        Data containing time-series of returns

    days_history : integer, default: 252
        Number of days to plot going back from last data point.

    log_scale : boolean, default: False
        Choose whether to plot cumulative returns on log scale y-axis.

    Returns
    ----------
    Historical cumulative return and maximum drawdown charts
    '''

    plt.figure(figsize=(17, 10), dpi=100)
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=4, colspan=1)
    ax1.plot((df.tail(days_history)+1).cumprod(), linewidth=1.5)
    ax1.set_yscale('log') if log_scale else None
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}x'.format(y)))
    ax1.legend(labels=df.columns, loc='upper left')
    plt.title('Cumulative Performance', fontsize=13, fontweight='bold')

    ax2 = plt.subplot2grid((8, 1), (4, 0), rowspan=4, colspan=1, sharex=ax1)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.plot(df.tail(days_history).apply(lambda x: np.cumprod(x + 1) / np.maximum.accumulate(np.cumprod(x + 1))-1), linewidth=1.5)
    ax2.legend(labels=df.columns, loc='lower right')
    plt.title('Historical Drawdowns', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=4, w_pad=0.5, h_pad=2.0)


def plot_results(results_dict, thresholds=(0,0), days_history=252, plot_regression=None):
    '''Plots classifier/regressor performance over the speicified time period.
    
    Parameters
    ----------
    results_dict : dictionary
        Dictionary containing data in Pandas DataFrame format to be plotted.
        There should be three tables in the dictionary:
        'returns':       return stream of each classifier, inclusing the Control to compare them to
        'predictions':   regression predictions
        'positioning':   long/short predictions, integer. (usually -1 or 1)

            Example:
            reg_results_dict = {'returns':df_returns, 
                                'predictions':df_predictions,
                                'positioning':df_positioning}

    thresholds : tuple, integer or float, default: (0,0)
        Tuple of two numerical thresholds to engage on (short, long) side. Used only for plotting markers.

    days_history : integer, default: 252
        Number of days to plot going back from last data point.

    plot_regression : string, default: None
        Choose which regression to plot.

    Returns
    ----------
    Backtest performance plot in matplotlib.
    '''

    df_predictions = results_dict.get('predictions')
    df_returns = results_dict.get('returns')
    df_positioning = results_dict.get('positioning')
    labels = df_predictions.columns
    long_threshold = thresholds[0]
    short_threshold = thresholds[1]
    days_history = days_history
    chosen_regression = plot_regression if plot_regression is not None else labels[0]
    plt.figure(figsize=(20, 12), dpi=100)
    ax1 = plt.subplot2grid((16, 1), (0, 0), rowspan=4, colspan=1)
    ax1.plot(df_predictions.tail(days_history))
    plt.axhline(y=long_threshold)
    plt.axhline(y=short_threshold)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.legend(labels=labels)
    ax2 = plt.subplot2grid((16, 1), (4, 0), rowspan=4, colspan=1, sharex=ax1)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.bar(df_returns.tail(days_history).index, -df_returns['Control'].tail(days_history),
            color=['#76767a' if i == 0 else 'r' if ((i>0 and j<0) or (i<0 and j>0))
                   else 'g' for i, j in zip(df_positioning[chosen_regression].tail(days_history), -df_returns['Control'].tail(days_history))])
    ax3 = plt.subplot2grid((16, 1), (8, 0), rowspan=4, colspan=1, sharex=ax1)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax3.plot(np.cumprod(df_returns['Control'].tail(days_history) + 1)-1)
    ax3.plot(np.cumprod(-df_returns['Control'].tail(days_history) * df_positioning[chosen_regression].tail(days_history) + 1)-1)
    ax3.legend(labels=['Control', chosen_regression])
    ax4 = plt.subplot2grid((16, 1), (12, 0), rowspan=4, colspan=1, sharex=ax1)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax4.plot(df_returns.tail(days_history).apply(lambda x: np.cumprod(x + 1) / np.maximum.accumulate(np.cumprod(x + 1))-1))
    ax4.legend(labels=df_returns.columns)
    plt.tight_layout(pad=4, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------
# VIX futures functions
#------------------------------------------------------------------------
def get_contract_expiry(contract):
    contract_months = ['f','g','h','j','k','m','n','q','u','v','x','z',]
    contract_expiry = range(1,13)
    contract_key = dict(zip(contract_months,contract_expiry))

    my_match = re.match(r'^.x(.)([0-9]+)$', contract)
    month = contract_key[my_match.group(1)]
    year = my_match.group(2)
    if year == '6':
        year = '16'
    if year == '7':
        year = '17'
    if year == '8':
        year = '18'
    if year == '9':
        year = '19'
    year = int(year) % 1000 + 2000
    my_date = datetime.datetime(year, month, 1)
    for expiry in get_last_trading_days(my_date, my_date + datetime.timedelta(365), -1):
        if expiry > my_date:
            return expiry
    return 0

def get_last_trading_days(from_date, to_date, days_shift=0):
    
    if not isinstance(from_date, datetime.date):
        from_date = parse(from_date)
    if not isinstance(to_date, datetime.date):
        to_date = parse(to_date)

    from_year = from_date.year
    to_year = to_date.year
    
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            # Expiration Date: The Wednesday that is thirty days prior to the third Friday of the
            # calendar month immediately following the month in which the contract expires
            
            # Compute the dates for each week that overlaps the month
            c = calendar.monthcalendar(year, month)# + calendar.monthcalendar(year+1, month)
            first_week = c[0]
            third_week = c[2]
            fourth_week = c[3]
            # If there is a Friday in the first week, the third Friday
            # is in the third week.  Otherwise the third Wednesday must 
            # be in the fourth week.
            
            if first_week[calendar.FRIDAY]:
                date = third_week[calendar.FRIDAY]
            else:
                date = fourth_week[calendar.FRIDAY]

            my_date = datetime.datetime(year, month, date) + datetime.timedelta(-30 + days_shift)

            if my_date > from_date and my_date <= to_date:
                yield my_date

def get_contract_days_length(date, how='total'):
    '''Returns the number of workdays in a contract period for the specified 'date' based on VIX futures expiry definition below.
       Expiration Date: The Wednesday that is thirty days prior to the third Friday of the
       calendar month immediately following the month in which the contract expires
            
    Parameters
    ----------
    how : str, 'total' or 'remaining', default: 'total'
        'total' returns the total number of workdays within the contract period. 'remaining' returns the number of workdays remaining in the contract period.
    '''
    if not isinstance(date, datetime.date):
        date = parse(date)
    last_trading_days = get_last_trading_days(date - datetime.timedelta(365), date + datetime.timedelta(365*2), -1)
    next_expiry = next(y for y in sorted(last_trading_days, reverse=False) if y > date)
    last_trading_days = get_last_trading_days(date - datetime.timedelta(365), date + datetime.timedelta(365*2), -1)
    prior_expiry = next(y for y in sorted(last_trading_days, reverse=True) if y <= date)

    nyse_cal = USTradingCalendar()
    mkt_holidays = nyse_cal.holidays() #[dt.date() for dt in nyse_cal.holidays()]
    if how == 'total':
        days_in_contract = workdays.networkdays(prior_expiry, next_expiry, mkt_holidays) - 1
    elif how == 'remaining':
        days_in_contract = workdays.networkdays(date, next_expiry, mkt_holidays) - 1
    else:
        raise ValueError("Unknown parameter how='%s'. Use 'total' or 'remaining'." % how)
    return days_in_contract


def get_contract_weight(date):
    '''Returns the weight of the front contract vs. the second contract, given the date based on VIX futures expiry definition below.
       Expiration Date: The Wednesday that is thirty days prior to the third Friday of the
       calendar month immediately following the month in which the contract expires
    
    Parameters
    ----------
    date : str or datetime
        If string, it gets parsed into a datetime object.
    '''

    weight = get_contract_days_length(date, how='remaining') / get_contract_days_length(date, how='total')
    return weight


def get_vix_futures_prices(start_date, end_date=datetime.datetime.today()):
    if not isinstance(start_date, datetime.date):
        start_date = parse(start_date)
    if not isinstance(end_date, datetime.date):
        end_date = parse(end_date)
        
    contract_months = ['f','g','h','j','k','m','n','q','u','v','x','z',]
    contract_expiry = range(1,13)
    contract_key = dict(zip(contract_expiry, contract_months))
    tickers = []
    for expiry in get_last_trading_days(start_date, end_date + datetime.timedelta(260), -1):
        if expiry > datetime.datetime.today() + datetime.timedelta(260):
            break
        elif expiry.year >= datetime.datetime.today().year:
            year = '{0:1d}'.format(expiry.year%2010)
        else:
            year = '{0:02d}'.format(expiry.year%1000)

        tickers.append('ux' + contract_key[expiry.month] + year)
        
    bb_tickers = [ticker + ' index' for ticker in tickers]
    start_date, end_date = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

    con = pdblp.BCon(debug=False)
    con.start()
    df_tmp1 = con.bdh(bb_tickers, 'px_last', start_date=start_date, end_date=end_date)
    con.stop()
    df_tmp1 = df_tmp1.xs('px_last', axis=1, level=1)
    df_tmp1.columns = [column.replace(' index','') for column in df_tmp1.columns]
    vix_contracts = df_tmp1[tickers].dropna(axis=1, how='all')
    vix_contracts.index = pd.to_datetime(vix_contracts.index)
    vix_contracts.columns = [get_contract_expiry(column) for column in vix_contracts.columns]
    vix_contracts.columns = pd.to_datetime(vix_contracts.columns)
    vix_contracts.index = pd.to_datetime(vix_contracts.index)
    
    vix_rolling_contracts = pd.DataFrame()
    ux1_date = vix_contracts.columns[0]
    ux2_date = vix_contracts.columns[1]
    ux3_date = vix_contracts.columns[2]
    ux4_date = vix_contracts.columns[3]
    ux5_date = vix_contracts.columns[4]
    ux6_date = vix_contracts.columns[5]
    ux7_date = vix_contracts.columns[6]
    for date in vix_contracts.index:
        if ux1_date <= date:
            for i, column in enumerate(vix_contracts.columns):
                if column > date:
                    ux1_date = ux2_date
                    if i < len(vix_contracts.columns) - 1:
                        ux2_date = vix_contracts.columns[i + 1]
                        ux3_date = vix_contracts.columns[i + 2]
                        ux4_date = vix_contracts.columns[i + 3]
                        ux5_date = vix_contracts.columns[i + 4]
                        ux6_date = vix_contracts.columns[i + 5]
                        ux7_date = vix_contracts.columns[i + 6]
                    break
        vix_rolling_contracts.loc[date, 'ux1'] = vix_contracts.loc[date, ux1_date]
        vix_rolling_contracts.loc[date, 'ux2'] = vix_contracts.loc[date, ux2_date]
        vix_rolling_contracts.loc[date, 'ux3'] = vix_contracts.loc[date, ux3_date]
        vix_rolling_contracts.loc[date, 'ux4'] = vix_contracts.loc[date, ux4_date]
        vix_rolling_contracts.loc[date, 'ux5'] = vix_contracts.loc[date, ux5_date]
        vix_rolling_contracts.loc[date, 'ux6'] = vix_contracts.loc[date, ux6_date]
        vix_rolling_contracts.loc[date, 'ux7'] = vix_contracts.loc[date, ux7_date]
    vix_rolling_contracts.index = pd.to_datetime(vix_rolling_contracts.index)
    return vix_rolling_contracts


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """

    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

class BayesianSearchCV():
    '''
    Cross validated hyperparemeter search class, similar to GridSearchCV, but utilizing
    Bayesian optimization via a gaussian process.

    Important members are fit, predict.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over a parameter grid.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_dict : dict of tuples
        Dictionary with parameters names (string) as keys and tuples of 
        parameter range bounds to pass to the bayesian optimizer 
        to try as values.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    '''
    def __init__(self, estimator, param_dict, scoring=None, n_jobs=1, cv=None, cv_exclude_first=0.0, verbose=0):
        self.estimator = estimator
        self.param_dict = param_dict
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.cv_exclude_first = cv_exclude_first
        self.verbose = verbose
        self.bayesian_optimizer = BayesianOptimization(self._evaluate, self.param_dict, verbose=verbose)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        _check_param_grid(self.param_dict)

    def _evaluate(self, **kwargs):
        if hasattr(self.estimator, 'hidden_layer_sizes'):
            hl_number = kwargs.pop('hidden_layers', None)
            hl_sizes = kwargs.pop('hidden_layer_sizes', None)
            kwargs['hidden_layer_sizes'] = (int(hl_sizes),) * int(hl_number)

        estimator = self.estimator.set_params(**kwargs)
        cv = check_cv(self.cv, self.y, classifier=is_classifier(self.estimator))
        scores = cross_val_score(estimator, self.X, self.y, scoring=self.scoring, cv=cv)
        return np.mean(scores[int(self.cv_exclude_first * self.cv.n_splits):])

    def fit(self, X, y=None, init_points=4, n_iter=10, acq='ucb', kappa=2.576, xi=0.0, gp_params={}):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        init_points : int, default=4
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        n_iter : int, default=10
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        acq : str, 'ucb', 'ei' or 'poi', default='ucb'
            Acquisition function to be used, defaults to Upper Confidence Bound.

        gp_params:
             default:
             alpha=1e-10, copy_X_train=True,
             kernel=Matern(length_scale=1, nu=1.5),
             n_restarts_optimizer=25, normalize_y=False,
             optimizer='fmin_l_bfgs_b', random_state=None

            Parameters to be passed to the Scikit-learn Gaussian Process object
        """
        self.X = X
        self.y = y
        self.bayesian_optimizer.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, **gp_params)
        
        self.cv_results_ = self.bayesian_optimizer.res

    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['max']['max_params']

    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['max']['max_val']

#------------------------------------------------------------------------
# Label Generation functions -- DEPRECATED
#------------------------------------------------------------------------
def generate_labels(dataset, method='simple', threshold=0.00):
    n=5
    
    if method == 'simple':
        # future return/classifier build
        rolling_days_test = range(1,n)
        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(lambda x: -1 if x < threshold else 1)
    
    elif method == 'log':
        rolling_days_test = range(1,n)
        def create_labels(x):
            if abs(x) > threshold*(1+np.log(days)):
                              return np.sign(x)*1
            else: return -1

        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(create_labels)

    elif method == 'regressor':
        # future return/classifier build
        rolling_days_test = range(1,n)
        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn']

    elif method == 'loglong':
        # future return/classifier build
        rolling_days_test = range(1,n)
        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(lambda x: 1 if x > threshold*(1+np.log(days)) else -1)

    elif method == 'ewm':
        # future return/classifier build - Exponential Moving Average
        rolling_days_test = range(1,n)

        for days in rolling_days_test:
            s = dataset['vxx equity_pct'].sort_index(ascending=False, inplace=False)
            s = s.ewm(span=days).mean().sort_index(ascending=True, inplace=False)
            dataset[str(days) + 'd_rtn'] = s.shift(-days)# .rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(lambda x: np.sign(x))

    elif method == 'multi_short':
        # future return/classifier build
        rolling_days_test = range(1,n)
        def create_labels(x):
            if x > 0.02:
                return 1
            elif -0.05 < x < 0.02:
                return -1
            elif x < -0.05:
                return -2
            else: return 0

        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(create_labels)

    elif method == 'multi':
        # future return/classifier build
        rolling_days_test = range(1,n)
        def create_labels(x):
            if x < -0.04:
                return -3
            elif x < -0.02:
                return -2
            elif x < 0.00:
                return -1
            elif x < 0.02:
                return 1
            elif x < 0.04:
                return 2
            else:
                return 3

        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(create_labels)

    elif method == 'median':
        # future return/classifier build
        rolling_days_test = range(1,n)
        def create_labels(x):
            if x > threshold:
                return 1
            else: return -1

        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = dataset[str(days) + 'd_rtn'].apply(create_labels)

    #-----------------------------RANDOM-DUMMY-LABEL-----------------------------------------------------
    elif method == 'dummy':
        rolling_days_test = range(1,n)
        for days in rolling_days_test:
            dataset[str(days) + 'd_rtn'] = (dataset['vxx equity_pct']).rolling(days).apply(lambda x: np.prod(x+1)-1).shift(-days)
            dataset[str(days) + 'd_rtn_flag'] = np.random.randint(-1,2, size=len(dataset))
            
    else:
        print('Error: bad method provided.')
        sys.exit()

    return dataset
