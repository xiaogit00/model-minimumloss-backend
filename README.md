# model-minimumloss-backend
To install and start: 

`pip install -r requirements.txt`  
`python app.py`

In dev mode, hot reload:    
`uvicorn app:app --reload`

### Project structure. 
This project uses file structures to generate backend routes and data. 

Essentially, `app.py` contains all the routes. The functions to pull data are defined in `services/file_db`. It uses a file based DB system, which pulls DB records from the `/models` folder. 

Each folder in the `/models` directory contains a model with the slug (e.g. `1_MNISTFashion_shallowNN_18FEB26`)

The `/GET models` route iterates through this models folder to get all the models. So this folder is an in-repo DB. 

### File rendering requirements. 
Each `model` folder **expects** the following files, which will be rendered in the frontend:
- DATASET.md (a write up of the dataset used for training)
- MODEL_DEFINITION.py (the Pytorch model implementation)
- TRAINING_CODE.py (the actual code used for training)
- EVAL_RESULTS.log (the logs generated from training)
- metadata.json (used for links to blogpost + tags)

Each of these are rendered in different pages on the frontend. 

Since these files are required for each model, the model generation code generates these for each model. 

To generate a new model, you run: `python3 generate_model.py`

### Using trained models for inference
The idea ultimately is to be able to serve the models themselves and use them for inference. Each model will be exported into the `models/modelSlug/exports` folder, and be exposed as a route for the frontend to be able to use it for inference. 