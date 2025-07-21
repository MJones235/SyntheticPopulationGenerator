from .base import HouseholdCompositionClassifier
from typing import Dict
import pandas as pd


class UKHouseholdCompositionClassifier(HouseholdCompositionClassifier):
    def get_name(self):
        return 'uk_census'

    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str = "relationship") -> Dict[str, float]:
        label_map = self.label_map()

        # Compute household-level classifications
        household_labels = synthetic_df.groupby("household_id").apply(
            lambda x: self.classify_household_structure(x, relationship_col)
        )
        synthetic_counts = household_labels.value_counts(normalize=True) * 100

        # Map internal labels to census categories
        synthetic_counts.index = synthetic_counts.index.map(lambda x: label_map.get(x, x))
        return synthetic_counts.to_dict()

    def classify_household_structure(self, group: pd.DataFrame, relationship_col: str = "relationship") -> str:
        n = len(group)
        roles = set(group[relationship_col])
        head = group[group[relationship_col] == "Head"]
        children = group[group[relationship_col] == "Child"]
        parents = group[group[relationship_col] == "Parent"]
        siblings = group[group[relationship_col] == "Sibling"]

        if head.empty:
            return "No head of household"

        if n == 1:
            return (
                "One-person household: Aged 66 years and over"
                if head.iloc[0]["age"] >= 66
                else "One-person household: Other"
            )

        allowed_family_roles = {"Head", "Partner", "Spouse", "Child"}
        if roles.issubset(allowed_family_roles):
            if "Partner" in roles or "Spouse" in roles:
                if children.empty:
                    return "Single family household: Couple family household: No children"
                elif any(children["age"] < 18):
                    return "Single family household: Couple family household: Dependent children"
                else:
                    return "Single family household: Couple family household: All children non-dependent"
            else:
                return "Single family household: Lone parent household"

        allowed_reverse_family_roles = {"Head", "Parent", "Sibling"}
        if roles.issubset(allowed_reverse_family_roles):
            parent_count = len(parents)
            head_age = head.iloc[0]["age"]
            sibling_ages = siblings["age"].tolist()

            if parent_count == 1:
                return "Single family household: Lone parent household"
            elif parent_count == 2:
                all_ages = [head_age] + sibling_ages
                if any(age < 18 for age in all_ages):
                    return "Single family household: Couple family household: Dependent children"
                else:
                    return "Single family household: Couple family household: All children non-dependent"
            else:
                return "Other household types"

        return "Other household types"
    
    def get_label_order(self):
        return [
            "One-person aged <66 years",
            "One-person aged 66+ years",
            "Lone parent",
            "Couple",
            "Couple with dependent children",
            "Couple with non-dependent children",
            "Other",
        ]

    def label_map(self):
        return {
            "One-person household: Aged 66 years and over": "One-person aged 66+ years",
            "One-person household: Other": "One-person aged <66 years",
            "Single family household: Lone parent household": "Lone parent",
            "Single family household: Couple family household: No children": "Couple",
            "Single family household: Couple family household: Dependent children": "Couple with dependent children",
            "Single family household: Couple family household: All children non-dependent": "Couple with non-dependent children",
            "Other household types": "Other",
        }