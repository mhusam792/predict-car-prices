from typing import Optional
from pydantic import BaseModel, model_validator


class CarFeatures(BaseModel):
    mileage: Optional[float] = None
    tax: Optional[float] = None
    mpg: Optional[float] = None
    engineSize: Optional[float] = None

    model: Optional[str] = None
    transmission: Optional[str] = None
    fuelType: Optional[str] = None
    Make: Optional[str] = None

    year: Optional[float] = None

    @model_validator(mode="after")
    def check_null_limit(self):
        null_count = sum(1 for v in self.model_dump().values() if v is None)

        if null_count > 4:
            raise ValueError("Please provide at least 5 fields to get a prediction.")

        return self
