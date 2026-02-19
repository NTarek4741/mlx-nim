from fastapi import FastAPI
import logging

app = FastAPI()

@app.post('/api/generate')
async def generate():
    pass

@app.post('/api/chat')
async def generate():
    pass

@app.post('/api/embed')
async def embed():
    pass

@app.post('/api/audio/tts')
async def embed():
    pass

@app.post('/api/audio/stt')
async def embed():
    pass

@app.post('/api/realtime')
async def embed():
    pass

@app.get('/api/tags')
async def tags():
    pass

@app.get('/api/ps')
async def tags():
    pass

@app.post('/api/show')
async def show():
    pass

@app.post('/api/create')
async def create():
    pass

@app.post('/api/copy')
async def create():
    pass

@app.post('/api/pull')
async def pull():
    pass


@app.post('/api/push')
async def push():
    pass

@app.post('/api/delete')
async def push():
    pass

@app.post('/api/version')
async def push():
    pass

@app.post('/v1/messages')
async def message():
    pass

@app.post('/v1/chat/completions')
async def chat_completions():
    pass