import requests
import json
from main.model_manager import ModelManager
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
import pathlib
import uvicorn


app = FastAPI()
app.mount("/img", StaticFiles(directory=pathlib.Path().cwd() / 'img'), name="img")
mod_manager = ModelManager()


def load_data():
    url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        print("Data loaded successfully")
    else:
        print("Data loading error")
        data = []
    df = pd.DataFrame(data[1:], columns=data[0])
    df[['density','speed','temperature']] = df[['density','speed','temperature']].astype(np.float32)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    max_date = df['time_tag'].max()
    min_date = max_date-pd.Timedelta(3, 'D')
    all_dates = pd.date_range(start=min_date, end=max_date, freq='T') # 'T' - minutely frequency
    all_dates_df = pd.DataFrame(all_dates, columns=['time_tag'])
    merged_df = pd.merge(all_dates_df, df, on='time_tag', how='left')
    three_days_ago = df['time_tag'].max() - pd.Timedelta(days=3)
    merged_df = merged_df[merged_df['time_tag'] > three_days_ago]
    merged_df = merged_df.infer_objects()
    merged_df.interpolate(method='linear', inplace=True)
    merged_df = merged_df.bfill()
    X = merged_df['density'].to_numpy()
    return X, merged_df['time_tag']


@app.on_event("startup")
def startup_event():
    mod_manager.load_model()


@app.get("/predict_event", response_class=HTMLResponse)
async def predict_event(model: CatBoostClassifier = Depends(mod_manager.get_model)):
    try:
        X, time_X = load_data()
        proba = model.predict_proba(X)
        prediction = model.predict(X)
        event_predict = "Storm possible" if prediction > 0.5 else "Storm unlikely"
        fig_X = go.Figure([go.Scatter(x=time_X, y=X, mode='lines')])
        fig_X.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            modebar_color='white',
            modebar_activecolor='green',
        )
        fig_X.update_traces(line=dict(color='rgba(0, 255, 0, 0.5)'))
        graph_X_html = pio.to_html(fig_X, full_html=False)

        return f'''<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            body {{
            background-color: white;
            color: black;
             }}
                .small-rectangle {{
                    width: 70%;
                    height: auto;
                    background-color: black;
                    color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    position: relative;
                    top: 20px;
                    left: 20px;
                    margin-top: 70px;
                }}
                .top-right-image {{
                position: absolute;
                top: 10px;
                right: 10px;
                height: 100px;
                }}
                .graph-container {{
                margin-top: 50px;
                }}
    </style>
            </style>
        </head>
        <body>
            <div class="small-rectangle">
                <img src="/img/cosmopulse_logo.png" class="top-right-image">
                <div>{event_predict}</div>
                <div>Probability of storm - {proba[1]}</div>
                <div class="graph-container">{graph_X_html}</div>
            </div>
        </body>
        </html>'''
    except Exception as e:
        return f"An error occurred: {e}"

