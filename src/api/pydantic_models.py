from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: int
    PricingStrategy: str
    SubscriptionId: str
    Feature1: Optional[float] = None
    Feature2: Optional[float] = None
    # Add all necessary feature fields matching your model input here

class RiskPredictionResponse(BaseModel):
    risk_probability: float



