# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predictor_api import make_prediction

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

@app.get("/api.json")
async def suggestion():
    return make_prediction()
