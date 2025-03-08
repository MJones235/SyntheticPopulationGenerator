import jsonschema

def validate_household(household_data: dict) -> bool:
    """Validates household data against the predefined schema."""
    from src.file_loaders.schema_loader import load_schema
    schema = load_schema("household_schema.json")
    try:
        jsonschema.validate(instance=household_data, schema=schema)
        return True 
    except jsonschema.exceptions.ValidationError as e:
        print(f"❌ Schema Validation Error: {e}")
        return False
