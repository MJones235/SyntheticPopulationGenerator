def is_number(value):
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    try:
        cleaned = value.replace(",", "").replace("_", "").strip()
        float(cleaned)
        return True
    except ValueError:
        return False
