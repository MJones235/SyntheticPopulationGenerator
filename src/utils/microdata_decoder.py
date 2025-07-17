from typing import Any, Dict
import pandas as pd


def decode_age(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Aged 15 years or under",
        2: "Aged 16 to 24 years",
        3: "Aged 25 to 34 years",
        4: "Aged 35 to 49 years",
        5: "Aged 50 to 64 years",
        6: "Aged 65 years or over"
    }
    return mapping.get(code, "Unknown")

def decode_household_type(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "One-person",
        2: "Married or civil partnership couple",
        3: "Cohabiting couple",
        4: "Lone parent",
        5: "Multi-person"
    }
    return mapping.get(code, "Unknown")

def decode_occupation(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Managers, directors or senior officials",
        2: "Professional occupations",
        3: "Associate professional and technical occupations",
        4: "Administrative and secretarial occupations",
        5: "Skilled trades occupations",
        6: "Caring, leisure and other service occupations",
        7: "Sales and customer service occupations",
        8: "Process, plant and machine operatives",
        9: "Elementary occupations"
    }
    return mapping.get(code, "Unknown")

def decode_sex(code: int) -> str:
    mapping = { 
        -8: "Does not apply",
        1: "Female",
        2: "Male"
    }
    return mapping.get(code, "Unknown")

def decode_economic_activity(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Employed",
        2: "Unemployed",
        3: "Economically inactive",
    }
    return mapping.get(code, "Unknown")

def decode_industry(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Agriculture, energy and water",
        2: "Manufacturing",
        3: "Construction",
        4: "Distribution, hotels and restaurants",
        5: "Transport and communication",
        6: "Financial, real estate, professional and administrative activities",
        7: "Public administration, education and health",
        8: "Other",
    }
    return mapping.get(code, "Unknown")

def decode_household_size(code: int) -> str:
    mapping = {
        0: "0 person",
        1: "1 person",
        2: "2 person",
        3: "3 person",
        4: "4 person",
        5: "5 person",
        6: "6 person",
        7: "7 person",
        8: "8 or more person",
    }
    return mapping.get(code, "Unknown")

def decode_adults_and_children(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "One person",
        2: "One adult and one or more children",
        3: "Two adults and no children",
        4: "Two adults and one or two children",
        5: "Two adults and three or more children",
        6: "Three or more adults and one or more children",
        7: "Three or more adults and no children",
    }
    return mapping.get(code, "Unknown")


def convert_microdata_row(row: pd.Series) -> str:
    paragraph = f"This anchor person is a {decode_sex(row["sex"]).lower()} {decode_age(row["resident_age_6a"]).lower()}. They live in a {decode_household_size(row["hh_size_9a"])} ({decode_household_type(row["hh_families_type_6a"]).lower()}) household."
    if row["hh_size_9a"] > 1:
        paragraph += f" The household contains {decode_adults_and_children(row['hh_adults_and_children_8m']).lower()}."

    econ_status = decode_economic_activity(row["economic_activity_status_4a"]).lower()
    occupation = decode_occupation(row["occupation_10a"])
    industry = decode_industry(row["industry_9a"])

    if econ_status == "employed":
        occupation_text = f"in an occupation categorised as {occupation.lower()}" if "does not apply" not in occupation.lower() else "in an unspecified role"
        industry_text = f"in the {industry} sector" if ("does not apply" not in industry.lower() and "other" not in industry.lower()) else ""
        paragraph += f" They are currently employed {occupation_text}"
        if industry_text:
            paragraph += f" {industry_text}"
        paragraph += "."
    elif "unemployed" in econ_status:
        paragraph += " They are currently unemployed and seeking work."
    elif "inactive" in econ_status:
        paragraph += " They are economically inactive."
    else:
        paragraph += f" Their economic status is: {econ_status}."

    return paragraph
