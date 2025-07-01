from typing import Any, Dict
import pandas as pd


def decode_age(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Aged 15 years and under",
        2: "Aged 16 to 24 years",
        3: "Aged 25 to 34 years",
        4: "Aged 35 to 44 years",
        5: "Aged 45 to 54 years",
        6: "Aged 55 to 64 years",
        7: "Aged 65 years and over"
    }
    return mapping.get(code, "Unknown")

def decode_household_type(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "One-person household",
        2: "Married or civil partnership couple household",
        3: "Cohabiting couple household",
        4: "Lone parent household",
        5: "Multi-person household"
    }
    return mapping.get(code, "Unknown")

def decode_ethnicity(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Asian, Asian British or Asian Welsh",
        2: "Black, Black British, Black Welsh, Caribbean or African",
        3: "Mixed or Multiple ethnic groups",
        4: "White",
        5: "Other ethnic group"
    }
    return mapping.get(code, "Unknown")

def decode_health(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Very good health",
        2: "Good health",
        3: "Fair health",
        4: "Bad health",
        5: "Very bad health"
    }
    return mapping.get(code, "Unknown")

def decode_partnership_status(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Never married and never registered a civil partnership",
        2: "Married or in a registered civil partnership",
        3: "Separated, but still legally married or still legally in a civil partnership",
        4: "Divorced or civil partnership dissolved",
        5: "Widowed or surviving civil partnership partner"
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

def decode_religion(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "No religion",
        2: "Christian",
        3: "Buddhist",
        4: "Hindu",
        5: "Jewish",
        6: "Muslim",
        7: "Sikh",
        8: "Other religion",
        9: "Not answered"
    }
    return mapping.get(code, "Unknown")

def decode_residence_type(code: int) -> str:
    mapping = {
        1: "Lives in a household",
        2: "Lives in a communal establishment",
    }
    return mapping.get(code, "Unknown")

def decode_sex(code: int) -> str:
    mapping = { 
        -8: "Does not apply",
        1: "Female",
        2: "Male"
    }
    return mapping.get(code, "Unknown")

def decode_country_of_birth(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Born in the UK",
        2: "Born outside the UK",
    }
    return mapping.get(code, "Unknown")

def decode_economic_activity(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Economically active (excluding full-time students): In employment: Employee",
        2: "Economically active (excluding full-time students): In employment: Self-employed",
        3: "Economically active (excluding full-time students): In employment: Unemployed: Seeking work or waiting",
        4: "Economically active and a full-time student",
        5: "Economically inactive: Retired",
        6: "Economically inactive: Student",
        7: "Economically inactive: Looking after home or family",
        8: "Economically inactive: Long-term sick or disabled",
        9: "Economically inactive: Other",
    }
    return mapping.get(code, "Unknown")

def decode_industry(code: int) -> str:
    mapping = {
        -8: "Does not apply",
        1: "Agriculture, forestry and fishing",
        2: "Manufacturing",
        3: "Energy and water",
        4: "Construction",
        5: "Distribution, hotels and restaurants",
        6: "Transport and communication",
        7: "Financial, real estate, professional and administrative activities",
        8: "Public administration, education and health",
        9: "Other",
    }
    return mapping.get(code, "Unknown")

def convert_microdata_row(row: pd.Series) -> str:
    paragraph = f"This individual is a {decode_sex(row["sex"])} {decode_age(row["resident_age_7d"]).lower()}. They live in a {decode_household_type(row["hh_families_type_6a"]).lower()}."

    econ_status = decode_economic_activity(row["economic_activity_status_10m"]).lower()
    occupation = decode_occupation(row["occupation_10a"])
    industry = decode_industry(row["industry_10a"])

    if "self-employed" in econ_status:
        industry_text = f"in the {industry} sector" if "does not apply" not in industry.lower() else ""
        paragraph += " They are self-employed"
        if industry_text:
            paragraph += f" {industry_text}"
        paragraph += "."
    elif "in employment" in econ_status:
        occupation_text = f"in an occupation categorised as {occupation.lower()}" if "does not apply" not in occupation.lower() else "in an unspecified role"
        industry_text = f"in the {industry} sector" if "does not apply" not in industry.lower() else ""
        paragraph += f" They are currently employed, {occupation_text}"
        if industry_text:
            paragraph += f" {industry_text}"
        paragraph += "."
    elif "unemployed" in econ_status:
        paragraph += " They are currently unemployed and seeking work."
    elif "student" in econ_status:
        paragraph += " They are a student."
    elif "retired" in econ_status:
        paragraph += " They are retired."
    elif "looking after home" in econ_status:
        paragraph += " They are not in paid employment and are looking after their home or family."
    elif "long-term sick" in econ_status:
        paragraph += " They are not in paid employment due to long-term sickness or disability."
    elif "inactive" in econ_status:
        paragraph += " They are economically inactive."
    else:
        paragraph += f" Their economic status is: {econ_status}."

    return paragraph
