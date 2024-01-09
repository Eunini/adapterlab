from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=3, max_length=3000)
    label: str = Field(min_length=1, max_length=64)
    group: str = Field(default="", max_length=160)


class DatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=80)
    description: str = Field(default="", max_length=500)
    license: str = Field(min_length=2, max_length=80)
    synthetic: bool = False
    records: list[Example] = Field(min_length=30, max_length=10_000)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    strategy: Literal["lora", "head-only", "full"] = "lora"
    rank: Literal[4, 8, 16] = 8
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.01)
    epochs: int = Field(default=4, ge=1, le=12)
    batch_size: Literal[4, 8, 16, 32] = 16
    max_length: int = Field(default=64, ge=16, le=192)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=80)
    dataset_id: str
    config: TrainConfig = Field(default_factory=TrainConfig)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=3, max_length=3000)
