from pydantic import constr, BaseModel, Field, ConstrainedStr
from typing import Literal, Optional, Union
import config

default_user_settings = config.default_user_settings

ret_curr = Literal["NOK", "SEK", "DKK", "USD", "GBP", "EUR"]
frequency = Literal["D", "W", "M", "Q", "A"]
universe = Field("all", min_length=2, max_length=20)
security_ticker = Field(None, min_length=1, max_length=10)
security_isin = Field(None, min_length=12, max_length=12)
security_name = Field(None, min_length=0, max_length=100)
security_weight = Field(None, gt=-1, le=1)
start_date = Field(None, min_length=10, max_length=10)
end_date = Field(None, min_length=10, max_length=10)
bm_id = Field(None, min_length=0, max_length=10)
ret_type = Literal["abs", "rel"]
factor_model = Literal["capm", "ff3", "ff3mom", "ff5", "ff5mom"]


class SettingsModel(BaseModel):
    date_format: Literal['yyyy-mm-dd', 'dd-mm-yyyy', 'yyyy.mm.dd', 'dd.mm.yyyy', 'yyyy/mm/dd', 'dd/mm/yyyy']
    thousand_sepearator: Literal[",", ".", " ", ""]
    decimal_sepearator: Literal[",", "."]

class PortfolioWeights(BaseModel):
    security_name: Optional[str] = security_name
    security_ticker: str = security_ticker
    security_isin: Optional[str] = security_isin
    security_weight: float = security_weight
    user_settings: Optional[SettingsModel] = default_user_settings

    
class PortfolioWeightsModel(BaseModel):
    fields: list[PortfolioWeights]
    user_settings: Optional[SettingsModel] = default_user_settings
