from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from model import classifier
import services.file_db as file_db
import os
import logging
from pathlib import Path
BASE_DIR = Path(__file__).parent
img_dir = BASE_DIR / "models/1_MNISTFashion_shallowNN_18FEB26/images"
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Image Classification API")

router = APIRouter()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@router.get("/")
async def root():
    return {"message": "Image Classification API is running"}
#------------------------------
#          Base Routes
#------------------------------
@router.get("/models")
async def models():
    res = file_db.get_models()
    return res

@router.get("/models/{model_slug}/definition")
async def model(model_slug):
    res = file_db.get_model_definition(model_slug)
    return res

@router.get("/models/{model_slug}/dataset")
async def model(model_slug):
    res = file_db.get_model_dataset(model_slug)
    return res

@router.get("/models/{model_slug}/training-code")
async def model(model_slug):
    res = file_db.get_model_training_code(model_slug)
    return res

@router.get("/models/{model_slug}/eval-results")
async def model(model_slug):
    res = file_db.get_model_eval_results(model_slug)
    return res

@router.get("/models/{model_slug}/loss-chart")
async def model(model_slug):
    res = file_db.get_model_loss_chart(model_slug)
    return res

@router.get("/models/{model_slug}/errors-chart")
async def model(model_slug):
    res = file_db.get_model_errors_chart(model_slug)
    return res

#------------------------------
#         Playground pages
#------------------------------
@router.get("/models/mnistfashion_test_images")
async def get_mnistfashion_images():
    files = os.listdir(img_dir)
    return {"images": [f"/mnistfashion_test_images/{f}" for f in files if f.endswith((".jpg", ".png", ".webp"))]}

# routes file
@router.get("/debug/images")  # → accessible at /api/debug/images
async def debug_images():
    return {
        "path": str(img_dir),
        "exists": img_dir.exists(),
        "files": os.listdir(img_dir) if img_dir.exists() else []
    }
#------------------------------
#         Troubleshooting
#------------------------------

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get classification predictions
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        
        # Get predictions
        predictions = classifier.predict(contents)
        
        logger.info(f"Successfully classified image: {file.filename}")
        
        return JSONResponse({
            "filename": file.filename,
            "predictions": predictions,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple images and get classifications
    """
    try:
        results = []
        for file in files:
            if file.content_type.startswith('image/'):
                contents = await file.read()
                predictions = classifier.predict(contents)
                results.append({
                    "filename": file.filename,
                    "predictions": predictions
                })
        
        return JSONResponse({
            "results": results,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api")
app.mount("/mnistfashion_test_images", StaticFiles(directory=img_dir), name="images")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)