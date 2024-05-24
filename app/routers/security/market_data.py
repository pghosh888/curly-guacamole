
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import constr
from app.components import is_authorized
from datetime import datetime, timedelta
from typing import Literal, Optional, Union, List


router = APIRouter(
    prefix="/security",
    tags=["Security market data"],
)
prefix = f"app/queries{router.prefix}/"

@router.get("/price/{ticker}")
def price(
    ticker: constr(min_length=1, max_length=10),
    auth = Depends(is_authorized),
):  
    #print(yf.download(ticker))
    return yf.download(ticker)[['Adj Close']].to_dict(orient="split")