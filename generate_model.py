#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
import re
from pathlib import Path

def get_valid_input(prompt, pattern=None, example=None):
    """Get user input with optional validation using regex pattern."""
    while True:
        user_input = input(prompt).strip()
        
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue
            
        if pattern and not re.match(pattern, user_input):
            if example:
                print(f"Invalid format. Example: {example}")
            else:
                print("Invalid format. Please try again.")
            continue
            
        return user_input

def main():
    # Define paths
    source_folder = "utils/model_generation_template"
    models_folder = "models"
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist!")
        return
    BASE_CODE_DIR = Path(__file__).parent
    models_dir = BASE_CODE_DIR / "models"
    print(models_dir)
    folderNames = [x.name for x in models_dir.iterdir() if x.is_dir()]
    models = [folderName.split('_') for folderName in folderNames]
    latestId = len(models)
    newId = latestId + 1

    print("\n" + "="*50)
    print("GENERATE A NEW MODEL")
    print("="*50)
    
    # Get user inputs with validation
    print("\nPlease provide the following information:")
    
    
    # Dataset name (alphanumeric, spaces, hyphens, underscores)
    dataset_name = get_valid_input(
        "Dataset Name: ",
        pattern=r'^[a-zA-Z0-9_\-\s]+$',
        example="cifar10 or image-net"
    )
    
    # Model name (alphanumeric, spaces, hyphens, underscores)
    model_name = get_valid_input(
        "Model Name: ",
        pattern=r'^[a-zA-Z0-9_\-\s]+$',
        example="resnet50 or bert-base"
    )
    
    # Get today's date in DDMMMYY format (e.g., 16FEB26)
    today = datetime.now()
    date_formatted = today.strftime("%d%b%y").upper()
    
    # Clean up names (replace spaces with underscores)
    dataset_name_clean = dataset_name.replace(' ', '_')
    model_name_clean = model_name.replace(' ', '_')
    
    # Create new folder name
    new_folder_name = f"{newId}_{dataset_name_clean}_{model_name_clean}_{date_formatted}"
    destination_path = os.path.join(models_folder, new_folder_name)
    
    # Check if destination already exists
    if os.path.exists(destination_path):
        overwrite = input(f"\nWarning: Folder '{new_folder_name}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
        shutil.rmtree(destination_path)
    
    # Copy the folder
    try:
        shutil.copytree(source_folder, destination_path)
        print(f"\n✅ Successfully copied template to:")
        print(f"   {destination_path}")
        
        # Optional: Display summary
        print("\n" + "-"*30)
        print("SUMMARY")
        print("-"*30)
        print(f"ID: {newId}")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")
        print(f"Date: {date_formatted}")
        print(f"Folder name: {new_folder_name}")
        print("-"*30)
        
    except Exception as e:
        print(f"\n❌ Error copying folder: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")