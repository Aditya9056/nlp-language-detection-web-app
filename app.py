# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from Input_validator import InputValidator
import numpy as np
import pickle
import pandas as pd
import lang_detect

# 2. Create the app object
app = FastAPI()
pickle_in = open("LRModel.pckl", "rb")
classifier = pickle.load(pickle_in)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 3. Index route
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 4. Send language data route
@app.post("/lang_detect/")
async def detect(data: InputValidator, request: Request):
    print("lang_data  is ", data)
    data = data.dict()
    text = data["lang_data"]

    # Creates lang detect object
    detect_model = lang_detect.Lang_detector()
    # print(detect_model.detect(text))
    return detect_model.detect(text)
