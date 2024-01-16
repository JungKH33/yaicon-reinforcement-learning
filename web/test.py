# to run,
# uvicorn test:app --reload
# in terminal

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
from typing import List

app = FastAPI()

# Mount the static files directory for HTML and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2 for HTML templates
templates = Jinja2Templates(directory="templates")

# Define the Gomoku board
gomoku_board = [["" for _ in range(15)] for _ in range(15)]


@app.get("/", response_class=HTMLResponse)
async def read_item():
    return templates.TemplateResponse("index.html", {"request": "request"})


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = {"message": f"Client {client_id}: {data}"}
            await manager.send_message(message, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/make_move/{row}/{col}")
async def make_move(row: int, col: int):
    if 0 <= row < 15 and 0 <= col < 15 and gomoku_board[row][col] == "":
        gomoku_board[row][col] = "X"  # Assuming the player is always "X"
        return {"status": "success"}
    else:
        return {"status": "error", "message": "Invalid move"}


@app.get("/get_board")
async def get_board():
    return {"board": gomoku_board}