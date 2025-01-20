from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Intelligent Document Search"}

@app.post("/query")
async def query_document(query: str):
    # Add retrieval and GPT-4 response logic here
    return {"response": "Your refined answer"}
