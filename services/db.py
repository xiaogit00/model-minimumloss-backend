from fastapi import HTTPException
import logging
from pathlib import Path


def get_models() -> list:
    """
    Gets all the filenames in folders path and returns them in a list of tuples

    :return: Description
    :rtype: list of tuples

    Example: [('1', 'MNISTFashion', 'shallowNN', '18FEB26'), ('2', 'MNISTFashion', 'deepNN', '21FEB26')]
    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    models_dir = BASE_CODE_DIR / "models"
    folderNames = [x.name for x in models_dir.iterdir() if x.is_dir()]
    models = [folderName.split('_') for folderName in folderNames]
    for i, name in enumerate(folderNames):
        models[i].append(name)
    return models

def get_model_definition(model_slug) -> str:
    """
    Gets the model definition python file and return as text

    :return: Model Definition Python code
    :rtype: str

    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    model_dir = BASE_CODE_DIR / "models" / str(model_slug)
    try:
        # Join the base directory with requested path
        model_definition_file_path = (model_dir / 'MODEL_DEFINITION.py').resolve()
        
        # Verify the resolved path is still within the base directory
        if not str(model_definition_file_path).startswith(str(model_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists and is a file
        if not model_definition_file_path.exists() or not model_definition_file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read and return the file content
        with open(model_definition_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("This is reached")
        return content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_model_dataset(model_slug) -> str:
    """
    Gets the model dataset MD file

    :return: Model dataset MD file
    :rtype: str

    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    model_dir = BASE_CODE_DIR / "models" / str(model_slug)
    try:
        # Join the base directory with requested path
        model_definition_file_path = (model_dir / 'DATASET.md').resolve()
        
        # Verify the resolved path is still within the base directory
        if not str(model_definition_file_path).startswith(str(model_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists and is a file
        if not model_definition_file_path.exists() or not model_definition_file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read and return the file content
        with open(model_definition_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_model_training_code(model_slug) -> str:
    """
    Gets the model dataset MD file

    :return: Model dataset MD file
    :rtype: str

    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    model_dir = BASE_CODE_DIR / "models" / str(model_slug)
    try:
        # Join the base directory with requested path
        training_code_file_path = (model_dir / 'TRAINING_CODE.py').resolve()
        
        # Verify the resolved path is still within the base directory
        if not str(training_code_file_path).startswith(str(model_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists and is a file
        if not training_code_file_path.exists() or not training_code_file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read and return the file content
        with open(training_code_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_model_eval_results(model_slug) -> str:
    """
    Gets the model dataset MD file

    :return: Model dataset MD file
    :rtype: str

    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    model_dir = BASE_CODE_DIR / "models" / str(model_slug)
    try:
        # Join the base directory with requested path
        training_code_file_path = (model_dir / 'EVAL_RESULTS.log').resolve()
        
        # Verify the resolved path is still within the base directory
        if not str(training_code_file_path).startswith(str(model_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists and is a file
        if not training_code_file_path.exists() or not training_code_file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read and return the file content
        with open(training_code_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# def get_models():
#     """Get all models"""
#     try:
#         res = (
#             supabase.table('models')
#             .select("*")
#             .execute()
#         )
#         if not res.data:
#             logging.info(f"No models found")
#             return None
#         logging.info(f'Returning {res.data} from get_models function')
#         return res.data
#     except Exception as e: 
#         print("There's an issue getting models from supabase: ", e)

def delete_models():
    try:
        res = (
            supabase.table('models')
            .delete()
            .neq("id", 0)
            .execute()
        )
        if not res.data:
            logging.info(f"No models found")
            return None
        logging.info(f'Deleted all models from models table.')
        return res.data
    except Exception as e: 
        print("There's an issue getting models from supabase: ", e)

def insert_model(modelData):
    try:
        newModel = {
            "slug": modelData['slug'],
            "name": modelData['name'],
            "description": modelData['description'],
            "model_architecture": modelData['model_architecture'],
            "reflections_url": modelData['reflections_url'],
            "dataset_description": modelData['dataset_description'],
            "dataset_url": modelData['dataset_url'],
            "training_code": modelData['training_code'],
            "model_code": modelData['model_code'],
            "tags": modelData['tags']
        }
        logging.info(f"New model object: {newModel}")
        res = (
            supabase.table('models')
            .insert(newModel)
            .execute()
        )
        if res.data:
            logging.info(f"Successfully inserted new model into models table.")
        return res
    except Exception as e: 
        print("There's an issue updating supabase table: ", e)