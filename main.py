import pandas as pd

from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, Body
from model.pipeline import predict_pipeline


app = FastAPI()


@app.get("/")
def root() -> str:
    return "Добро пожаловать на сервис"


@app.get("/ping")
def ping_get():
    return {"message": "OK"}


@app.post("/predict", summary="Predict")
def predict(
    vals: dict) -> int | dict:
    """Uploads samples and returns predictions as Json"""

    dict_vals = vals
    predicate = predict_pipeline(dict_vals)
    return predicate
