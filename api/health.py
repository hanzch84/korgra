from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/health")
def health_check():
    return JSONResponse({
        "status": "healthy",
        "sbert_loaded": True,
        "word2vec_loaded": True,
        "conceptnet_available": True
    })
