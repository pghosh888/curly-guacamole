
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import constr
from app.components import is_authorized
from datetime import datetime, timedelta
from typing import Literal, Optional, Union, List


router = APIRouter(
    prefix="/security/price",
    tags=["Security price"],
)
prefix = f"app/queries{router.prefix}/"

@router.get("/price/{ticker}")
def get_price(
    ticker: constr(min_length=1, max_length=10),
    auth = Depends(is_authorized),
):  
    
    return yf.download('AAPL')[ticker]['Close_Price'].to_dict(orient="split")