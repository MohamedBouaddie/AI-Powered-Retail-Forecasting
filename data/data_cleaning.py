import pandas as pd
import numpy as np


def clean_train(path="data/train.csv"):
    """
    Load and clean train.csv:
    Expects columns: date, store_nbr, family, sales, onpromotion
    """

    df = pd.read_csv(path, parse_dates=['date'])

    # Drop duplicates
    df = df.drop_duplicates(subset=['date', 'store_nbr', 'family'])

    #Checking for null
    df.loc[df[['date', 'store_nbr', 'family']].isnull().all(axis=1)]

    # Fill missing onpromotion with 0
    if 'onpromotion' in df.columns:
        df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)

    #Remove nan value raws 
    df = df.dropna(subset=['date', 'store_nbr', 'family', 'sales', 'onpromotion'])

    # Remove negative sales if any
    df = df[df['sales'] >= 0]

    # Optionally convert family to categorical
    df['family'] = df['family'].astype('category')
    
    # Sort for consistency
    df = df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)
    return df

def clean_test(path="data/test.csv"):
    """
    Load and clean test.csv:
    Expects columns: id, date, store_nbr, family, onpromotion
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # No sales column in test
    # Fill missing onpromotion with 0
    if 'onpromotion' in df.columns:
        df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)
    # Remove duplicates (just in case)
    df = df.drop_duplicates(subset=['id', 'date', 'store_nbr', 'family'])
    # Sort
    df = df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)
    return df

def clean_stores(path="data/stores.csv"):
    """
    Clean stores.csv:
    Expects columns: store_nbr, city, state, type, cluster
    """
    df = pd.read_csv(path)
    # Drop duplicates
    df = df.drop_duplicates(subset=['store_nbr'])
    return df

def clean_oil(path="data/oil.csv", date_min=None, date_max=None):
    """
    Clean oil.csv:
    Expects columns: date, dcoilwtico
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # Optionally expand to full date range
    if (date_min is not None) and (date_max is not None):
        full = pd.DataFrame({'date': pd.date_range(start=date_min, end=date_max, freq='D')})
        df = full.merge(df, how='left', on='date')
    # Interpolate missing oil prices
    if 'dcoilwtico' in df.columns:
        # Use linear interpolation
        df['dcoilwtico'] = df['dcoilwtico'].interpolate(method='linear')
        # For any still missing at edges, forward/backfill
        df['dcoilwtico'] = df['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
    return df

def clean_holidays(path="data/holidays_events.csv"):
    """
    Clean holidays_events.csv:
    Expects columns: date, type, locale, locale_name, transferred, maybe description
    Create a holiday flag.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # Drop duplicates
    df = df.drop_duplicates()
    # Normalize strings
    for col in ['type', 'locale', 'locale_name']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().fillna('none')
    # transferred: some are boolean or string — unify to boolean
    if 'transferred' in df.columns:
        # If string "True"/"False", convert; otherwise fill NaN as False
        df['transferred'] = df['transferred'].fillna(False)
        # Convert to bool if possible
        try:
            df['transferred'] = df['transferred'].astype(bool)
        except Exception:
            pass
    # Create holiday flag: “is_holiday = 1 if type is not work day and not transferred”
    df['is_holiday'] = ((df['type'] != 'work day') & (~df['transferred'])).astype(int)
    return df

def clean_transactions(path="data/transactions.csv"):
    """
    Clean transactions.csv:
    Expects columns: date, store_nbr, transactions
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # Drop duplicates
    df = df.drop_duplicates(subset=['date', 'store_nbr'])
    # Handle missing values by interpolation or fill
    if 'transactions' in df.columns:
        df['transactions'] = df['transactions'].interpolate(method='linear')
        # Fill ends
        df['transactions'] = df['transactions'].fillna(method='ffill').fillna(method='bfill')
    return df

def clean_sample_submission(path="data/sample_submission.csv"):
    """
    Clean sample_submission.csv:
    Expects columns: id, sales
    Mostly a template; ensure formatting
    """
    df = pd.read_csv(path)
    # Drop duplicates
    df = df.drop_duplicates(subset=['id'])
    # Ensure id is unique
    # Maybe ensure sales column exists (can be blank or zero)
    return df

def merge_all(train_df, test_df, stores_df, oil_df, holidays_df, transactions_df):
    """
    Merge cleaned dataframes into enriched train & test tables.
    Returns train_enriched, test_enriched
    """
    # Merge stores into train & test
    train = train_df.merge(stores_df, how='left', on='store_nbr')
    test = test_df.merge(stores_df, how='left', on='store_nbr')
    # Merge oil (by date)
    train = train.merge(oil_df, how='left', on='date')
    test = test.merge(oil_df, how='left', on='date')
    # Merge holidays (by date)
    # We only need holiday flags and maybe type, locale
    holidays_sub = holidays_df[['date', 'is_holiday', 'type', 'locale']]
    train = train.merge(holidays_sub, how='left', on='date')
    test = test.merge(holidays_sub, how='left', on='date')
    # Fill missing holiday info as non-holiday
    train['is_holiday'] = train['is_holiday'].fillna(0).astype(int)
    test['is_holiday'] = test['is_holiday'].fillna(0).astype(int)
    # Merge transactions
    train = train.merge(transactions_df, how='left', on=['date', 'store_nbr'])
    test = test.merge(transactions_df, how='left', on=['date', 'store_nbr'])
    # Fill transactions missing with 0 or interpolation
    train['transactions'] = train['transactions'].fillna(0)
    test['transactions'] = test['transactions'].fillna(0)
    return train, test

# if __name__ == "__main__":
    # Example usage: clean all and merge
train = clean_train("data/train.csv")
test = clean_test("data/test.csv")
stores = clean_stores("data/stores.csv")
oil = clean_oil("data/oil.csv", date_min=train['date'].min(), date_max=test['date'].max())
holidays = clean_holidays("data/holidays_events.csv")
trans = clean_transactions("data/transactions.csv")
train_en, test_en = merge_all(train, test, stores, oil, holidays, trans)
# Optionally save cleaned/merged
# train_en.to_csv("data/processed/train_enriched.csv", index=False, sep=';')
# test_en.to_csv("data/processed/test_enriched.csv", index=False, sep=';')


