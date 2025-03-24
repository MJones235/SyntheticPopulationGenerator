from src.repositories.db_manager import DBManager
import pandas as pd

class DashboardRepository:
    def __init__(self):
        self.db = DBManager()

    def get_estimations_with_metadata(self, variable: str) -> pd.DataFrame:
        query = """
            SELECT
            e.run_id,
            e.variable,
            e.location,
            e.category AS "BUA size classification",
            e.subcategory,
            e.ground_truth,
            e.trial_number,
            e.prediction,
            e.timestamp,
            m.model_name
        FROM estimations e
        LEFT JOIN estimation_metadata m ON e.run_id = m.run_id
        WHERE e.variable = ?
        """
        return pd.read_sql_query(query, self.db._connect(), params=(variable,))
