from fastapi import HTTPException
import logging
from pathlib import Path
import json


def get_models() -> list:
    """
    Gets all the filenames in folders path and returns them in a list of tuples

    :return: Description
    :rtype: list of lists

    Example: [('1', 'MNISTFashion', 'shallowNN', '18FEB26'), ('2', 'MNISTFashion', 'deepNN', '21FEB26')]
    """
    BASE_CODE_DIR = Path(__file__).parent.parent
    models_dir = BASE_CODE_DIR / "models"
    folderNames = [x.name for x in models_dir.iterdir() if x.is_dir()]
    models = [folderName.split('_') for folderName in folderNames]
    
    for i, name in enumerate(folderNames):
        metadata_file_path = (models_dir / name / 'metadata.json').resolve()
        with open(metadata_file_path, 'r') as file:
        # Use json.load() to parse the file content into a Python dictionary
            metadata = json.load(file)
            models[i].append(metadata['blog_link'])
            models[i].append(metadata['tags'])
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

