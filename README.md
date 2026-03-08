# NLP Language Detection Web App

Small end-to-end project for language detection using a custom NLP model exposed through a FastAPI service and consumed by a lightweight web app.

## What it does

- Accepts text input from the browser
- Sends the request to a FastAPI backend
- Predicts the language using a trained classification model
- Returns the detected language to the UI

## Supported languages

- English
- French
- Spanish
- Russian
- Chinese

## Stack

- Python
- FastAPI
- Scikit-learn based language classification workflow
- HTML, CSS, and JavaScript for the client

## Notes

- The training dataset and serialized model artifacts are not committed to this repository because of size constraints.
- The project can be extended to support more languages by retraining the model with additional data.

## Why it matters

This project is one of the more complete public examples of combining model training, API design, and a user-facing interface in a single workflow.
