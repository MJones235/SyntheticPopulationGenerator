import jsonschema
from src.services.file_service import FileService

def validate_household(household_data: dict) -> bool:
    """Validates household data against the predefined schema."""
    file_service = FileService()
    schema = file_service.load_schema("household_schema.json")
    try:
        jsonschema.validate(instance=household_data, schema=schema)
        return True 
    except jsonschema.exceptions.ValidationError as e:
        print(f"❌ Schema Validation Error: {e}")
        return False
