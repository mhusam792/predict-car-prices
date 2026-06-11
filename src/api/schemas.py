from typing import Optional
from pydantic import BaseModel


class CarFeatures(BaseModel):
    mileage: Optional[float] = None
    tax: Optional[float] = None
    mpg: Optional[float] = None
    engineSize: Optional[float] = None

    model: Optional[str] = None
    transmission: Optional[str] = None
    fuelType: Optional[str] = None
    Make: Optional[str] = None

    year: Optional[int] = None
