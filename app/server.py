from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.rag_chain import final_chain

# Initialize the FastAPI application
app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    """
    Redirect the root URL ("/") to the API documentation at "/docs".
    
    Returns:
        RedirectResponse: A response that redirects the client to the "/docs" endpoint.
    """
    return RedirectResponse("/docs")

# Add the RAG chain endpoint to the FastAPI application
add_routes(app, final_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application using Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
