from pydantic import BaseModel, Field


class CarFeatures(BaseModel):
    model: str
    year: int = Field(gt=1900)
    transmission: str
    mileage: float = Field(ge=0)
    fuelType: str
    tax: float = Field(ge=0)
    mpg: float = Field(gt=0)
    engineSize: float = Field(gt=0)
    Make: str


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    model_version: str
    model_alias: str
