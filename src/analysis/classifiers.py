import pandas as pd

def classify_household_structure(group: pd.DataFrame, relationship_col: str = "relationship") -> str:
    n = len(group)
    roles = set(group[relationship_col])
    head = group[group[relationship_col] == "Head"]
    children = group[group[relationship_col] == "Child"]

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


def household_type_labels():
    label_map = {
        "One-person household: Aged 66 years and over": "One-person aged 66+ years",
        "One-person household: Other": "One-person aged <66 years",
        "Single family household: Lone parent household": "Lone parent",
        "Single family household: Couple family household: No children": "Couple",
        "Single family household: Couple family household: Dependent children": "Couple with dependent children",
        "Single family household: Couple family household: All children non-dependent": "Couple with non-dependent children",
        "Other household types": "Other",
    }

    label_order = [
        "One-person aged <66 years",
        "One-person aged 66+ years",
        "Lone parent",
        "Couple",
        "Couple with dependent children",
        "Couple with non-dependent children",
        "Other",
    ]

    return label_map, label_order
