from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from typing import Optional, Dict, Any
import os
import json

from main import chain, vectorstore

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="Personal Knowledge Base API",
    description="""
    API for querying your personal knowledge base using RAG (Retrieval Augmented Generation).
    
    Available endpoints:
    - GET /: Chat interface
    - GET /status: Check knowledge base status
    - POST /query: Query your knowledge base
    """,
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are my key skills and experiences?"
            }
        }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface."""
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/query")
async def query_knowledge_base(query: Query):
    """Query your knowledge base."""
    try:
        # Create message
        message = HumanMessage(content=query.query)
        
        # Run the chain
        result = chain.invoke({
            "messages": [message]
        })
        
        # Get the agent outputs
        agent_outputs = result.get("agent_outputs", {})
        
        # Return formatted response
        return {
            "response": "\n".join([
                output for output in agent_outputs.values()
                if output is not None
            ])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get the current status of your knowledge base."""
    try:
        collection = vectorstore._collection
        return {
            "status": "ready",
            "documents_count": collection.count(),
            "collection_name": collection.name
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
