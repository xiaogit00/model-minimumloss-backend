import os
from services.supabase_client import get_supabase_client
from dotenv import load_dotenv
import logging
from typing import Optional
from datetime import datetime

load_dotenv()
supabase = get_supabase_client()

def truncate_models():
    try:
    # Option 1: Using RPC (requires defining a custom function in Supabase SQL Editor)
    # create or replace function truncate_models() returns void as $$
    # begin
    #   truncate table models restart identity;
    # end;
    # $$ language plpgsql;
    
        result = supabase.rpc("truncate_models").execute()
        print("Table truncated successfully")
    except Exception as e:
        print(f"Error: {e}")

def get_model(slug):
    """Get all models"""
    try:
        res = (
            supabase.table('models')
            .select("*")
            .eq('slug', slug)
            .execute()
        )
        if not res.data:
            logging.info(f"No models found")
            return None
        logging.info(f'Returning {res.data} from get_models function')
        return res.data
    except Exception as e: 
        print("There's an issue getting models from supabase: ", e)

def get_models():
    """Get all models"""
    try:
        res = (
            supabase.table('models')
            .select("*")
            .execute()
        )
        if not res.data:
            logging.info(f"No models found")
            return None
        logging.info(f'Returning {res.data} from get_models function')
        return res.data
    except Exception as e: 
        print("There's an issue getting models from supabase: ", e)

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