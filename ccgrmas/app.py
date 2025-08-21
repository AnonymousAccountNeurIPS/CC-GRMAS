from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ccgrmas.endpoints import graph, analytics, rag, spatial_prediction
import warnings
warnings.filterwarnings("ignore") 

app = FastAPI(
    title="CC-GRMAS API",
    description="Climate Change - Graph Risk Management and Analysis System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(graph.router)
app.include_router(analytics.router)
app.include_router(rag.router)
app.include_router(spatial_prediction.router)


@app.get("/")
def root():
    return {"message": "Welcome to CC-GRMAS API"}