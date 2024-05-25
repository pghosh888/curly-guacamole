import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import constr
from app.components import is_authorized, unpack_portfolio
from app.components.base_models import PortfolioWeightsModel
from datetime import datetime, timedelta
from typing import Literal, Optional, Union, List
from predictionrevisited import *

router = APIRouter(
    prefix="/portfolio",
    tags=["Portfolio analysis"],
)
prefix = f"app/queries{router.prefix}/"


@router.post("/analysis")
def portfolio_stress_test(
    base_model: PortfolioWeightsModel,
    auth=Depends(is_authorized),
):
    tickers, weights, weights_dict, names, isin = unpack_portfolio(base_model)
    price_df = {}
    for t in tickers:
        price_df[t] = yf.download(t)['Adj Close']