from .base import HouseholdCompositionClassifier
from typing import Dict
import pandas as pd


class UNHouseholdCompositionClassifier(HouseholdCompositionClassifier):
    def get_name(self):
        return 'un_global'

    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str = "relationship") -> Dict[str, float]:
        household_labels = synthetic_df.groupby("household_id").apply(lambda x: self.classify_household_structure(x, relationship_col))
        counts = household_labels.value_counts(normalize=True) * 100
        return counts.to_dict()

    def classify_household_structure(self, group: pd.DataFrame, relationship_col: str = "relationship") -> str:
        roles = group[relationship_col].tolist()
        n = len(roles)

        if n == 1:
            return "One-person"

        has_partner = any(r in {"Spouse", "Partner"} for r in roles)
        has_child = any(r == "Child" for r in roles)

        # Couple only
        if n == 2 and has_partner:
            return "Couple"

        # Nuclear couple with children — no other roles
        if has_partner and has_child and all(r in {"Head", "Spouse", "Partner", "Child"} for r in roles):
            return "Couple with children"

        # Lone parent with children only
        if not has_partner and has_child and all(r in {"Head", "Child"} for r in roles):
            return "Lone parent"

        # Reverse nuclear families (e.g., adult child is Head, living with parents and possibly siblings)
        if not has_partner and not has_child:
            reverse_roles = {"Head", "Parent", "Sibling"}
            if all(r in reverse_roles for r in roles):
                n_parents = sum(r == "Parent" for r in roles)
                if n_parents == 2:
                    return "Couple with children"
                elif n_parents == 1:
                    return "Lone parent"

        # Extended family: all members are relatives
        all_relatives = all(
            r in {
                "Head", "Spouse", "Partner", "Child", "Child-in-law",
                "Parent", "Sibling", "Sibling-in-law",
                "Grandchild", "Grandparent", "Aunt", "Uncle",
                "Nephew", "Niece", "Cousin"
            }
            for r in roles
        )
        if all_relatives:
            return "Extended family"

        return "Non-relatives"
    
    def get_label_order(self):
        return [
            "One-person",
            "Lone parent",
            "Couple",
            "Couple with children",
            "Extended family",
            "Non-relatives"
        ]