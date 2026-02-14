import os
from services.supabase_client import get_supabase_client
from dotenv import load_dotenv
import logging
from typing import Optional
from datetime import datetime

load_dotenv()
supabase = get_supabase_client()

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
