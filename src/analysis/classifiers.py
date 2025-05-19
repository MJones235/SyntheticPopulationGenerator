import pandas as pd

def classify_household_structure(group: pd.DataFrame) -> str:
    roles = set(group["relationship"])
    n_roles = len(roles)
    n = len(group)
    has_spouse = "Spouse" in roles
    has_child = "Child" in roles
    has_parent = "Parent" in roles
    has_grandchild = "Grandchild" in roles
    has_roommate = "Roommate" in roles

    head = group[group["relationship"] == "Head"]
    children = group[group["relationship"] == "Child"]
    parents = group[group["relationship"] == "Parent"]

    if (has_parent and has_child) or (has_grandchild and has_child) or (has_grandchild and has_parent):
        return "Multigenerational"
    
    if n == 1:
        return "Single Person"
    
    if has_roommate and n_roles == 2:
        return "Shared House"
    
    if has_spouse and n_roles == 2:
        return "Couple"
    
    if has_spouse and has_child and n_roles == 3:
        if any(children["age"] >= 21):
            return "Couple with Adult Child(ren)"
        return "Couple with Child(ren)"
    
    if has_child and n_roles == 2:
        if any(children["age"] >= 21):
            return "Single Parent with Adult Child(ren)"
        return "Single Parent with Child(ren)"
    
    print(roles, n_roles, has_child)
    return "Other"