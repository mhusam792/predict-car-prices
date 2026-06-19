# data_schema.py

from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    splitting_random_state: int = Field(ge=0)


class RareLabelConfig(BaseModel):
    tol: float = Field(gt=0, lt=1)
    variables: list[str]
