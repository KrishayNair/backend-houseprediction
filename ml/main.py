from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Home(BaseModel):
    avg_area_income: float
    avg_area_house_age: int
    avg_area_number_of_rooms: int
    avg_area_number_of_bedrooms: int
    area_population: int


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/predict")
async def predict(details: Home):
    df = pd.DataFrame([details.dict().values()], columns=details.dict().keys())
    print(df)
    pred = model.predict(df)
    return {
        "prediction": int(pred[0])
    }
