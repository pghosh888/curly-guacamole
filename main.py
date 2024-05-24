from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum


app = FastAPI()

### CORS  -to whitelist the incoming requests
origins = [
    "http://localhost:3000",
]


### Hello world
@app.get("/")
async def start_page():
    return {"swagger": "http://127.0.0.1:8000/docs"}

@app.get("/locale")
async def start_page():
    import locale

    return locale.locale_alias


### Handler
handler = Mangum(app)