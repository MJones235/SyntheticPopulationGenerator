import pandas as pd

def classify_household_structure(group: pd.DataFrame) -> str:
    n = len(group)
    roles = set(group["relationship"])
    head = group[group["relationship"] == "Head"]
    children = group[group["relationship"] == "Child"]

    if head.empty:
        return "No head of household"

    if n == 1:
        age = head.iloc[0]["age"]
        if age >= 66:
            return "One-person household: Aged 66 years and over"
        else:
            return "One-person household: Other"

    allowed_roles = {"Head", "Partner", "Child"}
    if not roles.issubset(allowed_roles):
        return "Other household types"

    if "Partner" in roles:
        if children.empty:
            return "Single family household: Couple family household: No children"
        elif any(children["age"] < 18):
            return "Single family household: Couple family household: Dependent children"
        else:
            return "Single family household: Couple family household: All children non-dependent"

    if "Child" in roles and "Partner" not in roles:
        return "Single family household: Lone parent household"

    return "Other household types"