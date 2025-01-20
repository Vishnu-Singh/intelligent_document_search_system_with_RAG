from fastapi import FastAPI
from pipeline import OpenAIPipeline
from configprocessor import ConfigProcessor

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Intelligent Document Search"}

@app.post("/query")
async def query_document(query: str):
    with ConfigProcessor("config.ini") as config:
        with OpenAIPipeline(config=config) as pipeline:
            result = pipeline.search_in_gpt4(query=query)
            # Add retrieval and GPT-4 response logic here
            return {"response": result}
