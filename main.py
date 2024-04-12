import pandas as pd

from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, Body
from model.pipeline import predict_pipeline

import wget

prefix = 'https://drive.google.com/uc?/export=download&id='

model_url = "https://drive.google.com/file/d/1Pk_qlDIYygNQzHNZuQiK-jU_tb0dSalT/view?usp=sharing"
model_file_id = model_url.split('/')[-2]
wget.download(prefix+model_file_id, out="/opt/render/project/src/model/RFClf.pkl")

vec_url = "https://drive.google.com/file/d/1O4HNdrjJPF2QNSDhlbmtKkqPEkVhlLv-/view?usp=sharing"
vec_file_id = vec_url.split('/')[-2]
wget.download(prefix+vec_file_id, out="/opt/render/project/src/model/CountVectorizer.pkl")


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
