import pandas as pd
import numpy as np
from data_cleaning import train_en

def add_calendar_features(df, date_col='date'):
    """
    Adds calendar / time-based features from a date column.
    df: DataFrame with a date column.
    Returns a DataFrame (a copy) with new features.
    """

    df = df.copy()
    # Ensure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df['dow'] = df[date_col].dt.dayofweek            # Monday=0 … Sunday=6
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    # cyclical encoding for dow

    return df

def add_lag_features(series, lags):
    """
    lags: list of integer lag values (e.g. [1,7,14,28])
    Returns DataFrame with columns lag_x for each lag.
    """
    df = pd.DataFrame({'y': series})
    for lag in lags:
        df[f'lag_{lag}'] = series.shift(lag)
    return df

def add_rolling_features(series, windows):
    """
    series: pd.Series
    windows: list of window sizes (e.g. [7,14,28])
    Returns DataFrame with rolling means, rolling stds.
    Note: shifts by 1 so current value not included.
    """
    df = pd.DataFrame({'y': series})
    for w in windows:
        df[f'roll_mean_{w}'] = series.shift(1).rolling(window=w).mean()
        df[f'roll_std_{w}'] = series.shift(1).rolling(window=w).std()
    return df

def add_diff_features(series, lags):
    """
    series: pd.Series
    lags: list of lags for difference (e.g. [1,7])
    Returns DataFrame with diff_x = y - y.shift(lag)
    """
    df = pd.DataFrame({'y': series})
    for lag in lags:
        df[f'diff_{lag}'] = series - series.shift(lag)
    return df

def build_feature_matrix(
    ts_series,
    date_index=None,
    onpromotion=None,
    is_holiday=None,
    transactions=None,
    lags=(1,7,14,28),
    rolling_windows=(7,14,28),
    diff_lags=(1,7)
):
    """
    Returns a DataFrame with features + target y.

    Parameters:
    - ts_series: pd.Series of target values (sales), indexed by date
    - date_index: pd.DatetimeIndex corresponding to ts_series
    - onpromotion: pd.Series aligned to same index (0/1), optional
    - is_holiday: pd.Series aligned (0/1), optional
    - transactions: pd.Series aligned, optional
    - lags, rolling_windows, diff_lags: tuples/lists of integers

    Returns:
    - DataFrame: index = dates, columns = features + 'y', rows with no NaNs dropped
    """
    # Basic frame
    df = pd.DataFrame({'y': ts_series}, index=date_index)

    # Calendar features
    df = df.reset_index().rename(columns={'index': 'date'})
    df = add_calendar_features(df, date_col='date')
    df = df.set_index('date')

    # Lag features
    lag_df = add_lag_features(ts_series, lags)
    df = df.join(lag_df.drop(columns=['y']), how='left')

    # Rolling features
    roll_df = add_rolling_features(ts_series, rolling_windows)
    df = df.join(roll_df.drop(columns=['y']), how='left')

    # Difference features
    diff_df = add_diff_features(ts_series, diff_lags)
    df = df.join(diff_df.drop(columns=['y']), how='left')

    # External / exogenous features
    if onpromotion is not None:
        df = df.join(onpromotion.rename('onpromotion'), how='left')
    if is_holiday is not None:
        df = df.join(is_holiday.rename('is_holiday'), how='left')
    if transactions is not None:
        df = df.join(transactions.rename('transactions'), how='left')

    # Drop rows with NaNs (these come from lags/roll features)
    df = df.dropna(subset=[col for col in df.columns if 'lag' in col or 'roll' in col])

    return df

train_en['date'] = pd.to_datetime(train_en['date'])
# Group by date (daily aggregation)
sales = train_en.groupby('date')['sales'].sum().sort_index()
onpromotion = train_en.groupby('date')['onpromotion'].mean().sort_index()
is_holiday = train_en.groupby('date')['is_holiday'].max().sort_index()
transactions = train_en.groupby('date')['transactions'].sum().sort_index()

# Get the index of dates
dates = sales.index

onpromotion = onpromotion.reindex(dates, fill_value=0)
is_holiday = is_holiday.reindex(dates, fill_value=0)
transactions = transactions.reindex(dates, fill_value=0)

df_feats = build_feature_matrix(
    ts_series=sales,
    date_index=dates,
    onpromotion=onpromotion,
    is_holiday=is_holiday,
    transactions=transactions
)

df_feats.to_csv("data/processed/features.csv", index=True, sep=';')