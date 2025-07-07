import os
import logging
import psutil
import uvicorn
import warnings
import json

warnings.filterwarnings(action='ignore')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.constant import ALLOW_ORIGINS, APP_HOST, APP_PORT
from src.utils.endpoint_filter import EndpointFilter

from main import move_2_players, move_3_players, move_2_players_honestly

from fastapi import Request
from fastapi.responses  import JSONResponse
from fastapi.encoders import jsonable_encoder
from prometheus_fastapi_instrumentator import Instrumentator

uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(EndpointFilter(path="/"))



app = FastAPI(
    title='TLMN_see_opponent_cards', 
    version='0.0.1',
    root_path='/TLMN-see-opponent-cards-bot',
    root_path_in_servers=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
async def read_root():
    return {"message": "Welcome to Justin!"}


@app.post("/move_2_players")
async def get_move_2_players(request_card: Request) -> JSONResponse:
    formated_request = await request_card.json()
    best_move = move_2_players(formated_request)

    return JSONResponse(
        content=jsonable_encoder(best_move),
        status_code=200
    )

@app.post("/move_2_players_honestly")
async def get_move_2_players_honestly(request_card: Request) -> JSONResponse:
    formated_request = await request_card.json()
    best_move = move_2_players_honestly(formated_request)

    return JSONResponse(
        content=jsonable_encoder(best_move),
        status_code=200
    )

@app.post("/move_3_players")
async def get_move_3_players(request_card: Request) -> JSONResponse:
    formated_request = await request_card.json()
    best_move = move_3_players(formated_request)

    return JSONResponse(
        content=jsonable_encoder(best_move),
        status_code=200
    )

if __name__ == "__main__":
    uvicorn.run(
        app='app:app', 
        host=APP_HOST, 
        port=APP_PORT, 
        log_level='info'
    )
