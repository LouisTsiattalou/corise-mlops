import os
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destination specified by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global classifier
    classifier = NewsCategoryClassifier()
    classifier.load(MODEL_PATH)
    logger.add(LOGS_OUTPUT_PATH)

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    logger.remove(LOGS_OUTPUT_PATH)
    os.remove(LOGS_OUTPUT_PATH)
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """

    start = datetime.now()

    model_input = {
        "source": request.source,
        "url": request.url,
        "title": request.title,
        "description": request.description,
    }

    scores = classifier.predict_proba(model_input)
    label = classifier.predict_label(model_input)

    prediction = {
        "scores": scores,
        "label": label,
    }

    end = datetime.now()

    logger.info({
        "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
        "request": model_input,
        "prediction": prediction,
        "latency": (end - start).total_seconds() * 1000,
    })

    return PredictResponse(scores=prediction["scores"], label=prediction["label"])


@app.get("/")
def read_root():
    return {"Hello": "World"}
