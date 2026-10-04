import csv
import json
import io
from pathlib import Path

def load_comments(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {filepath}")

    with open(path, "r", encoding="utf-8") as file:
        if path.suffix.lower() == '.json':
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("JSON data must be a list of dictionaries")
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
        elif path.suffix.lower() == '.csv':
            try:
                reader = csv.DictReader(file)
                return list(reader)
            except csv.Error as e:
                raise ValueError(f"Invalid CSV file: {e}")
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

def load_comments_from_string(csv_string):
    if not csv_string.strip():
        raise ValueError("CSV string is empty")
        
    try:
        reader = csv.DictReader(io.StringIO(csv_string))
        return list(reader)
    except csv.Error as e:
        raise ValueError(f"Invalid CSV string: {e}")

def load_comments_from_dicts(data: list[dict]):
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("All elements in the list must be dictionaries")
    return data

def save_comments(comments, filepath):
    if not isinstance(comments, list):
        raise ValueError("Comments must be a list of dictionaries")
    if not comments:
        raise ValueError("Comments list is empty")
        
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8", newline='') as file:
        if path.suffix.lower() == '.json':
            json.dump(comments, file, indent=2)
        elif path.suffix.lower() == '.csv':
            fieldnames = comments[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comments)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

def get_data_summary(comments):
    if not isinstance(comments, list):
        raise ValueError("Comments must be a list of dictionaries")
        
    platforms = set()
    brands = set()
    
    for comment in comments:
        if "platform" in comment:
            platforms.add(comment["platform"])
        if "brand" in comment:
            brands.add(comment["brand"])
            
    return {
        "count": len(comments),
        "platforms": list(platforms),
        "brands": list(brands)
    }