from src.repositories.db_manager import DBManager
import pandas as pd

class DashboardRepository:
    def __init__(self):
        self.db = DBManager()

    def get_estimations_with_metadata(self, variable: str) -> pd.DataFrame:
        query = """
            SELECT 
                m.variable,
                m.model_name,
                e.location,
                e.category AS "BUA size classification",
                e.ground_truth,
                e.prediction,
                e.trial_number
            FROM estimations e
            JOIN estimation_metadata m ON e.run_id = m.run_id
            WHERE m.variable = ?
        """
        return pd.read_sql_query(query, self.db._connect(), params=(variable,))
