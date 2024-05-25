from nqlib import read_sql, db_to_df
import pandas as pd
import config

#database = config.database

def unpack_portfolio(base_model):
    tickers = []
    weights_dict = {}
    weights = []
    names = []
    isin = []

    for x in range(len(base_model.fields)):
        ticker = base_model.fields[x].security_ticker
        tickers.append(ticker)
        weights.append(base_model.fields[x].security_weight)
        weights_dict[ticker] = base_model.fields[x].security_weight
        if base_model.fields[x].security_name is not None:
            names.append(base_model.fields[x].security_name)
        if base_model.fields[x].security_isin is not None:
            isin.append(base_model.fields[x].security_isin)

    return tickers, weights, weights_dict, names, isin