PALE_COLORS = [
    "background-color: rgba(255, 204, 204, 0.5)",  # Light Red
    "background-color: rgba(204, 229, 255, 0.5)",  # Light Blue
    "background-color: rgba(204, 255, 204, 0.5)",  # Light Green
    "background-color: rgba(255, 255, 204, 0.5)",  # Light Yellow
    "background-color: rgba(229, 204, 255, 0.5)"   # Light Purple
]


def assign_household_colors(df):
    """Assigns a pale background color to each household_id, cycling through 5 colors."""
    unique_households = df["household_id"].unique()
    
    # Cycle through the 5 colors
    household_colors = {
        household: PALE_COLORS[i % len(PALE_COLORS)] for i, household in enumerate(unique_households)
    }

    def apply_colors(row):
        return [household_colors[row["household_id"]]] * len(row)

    return df.style.apply(apply_colors, axis=1)
