import logging
import sqlite3

class DBManager:
    db_path = "data/outputs.sqlite"
    
    def __init__(self):
        self._connect().execute("PRAGMA foreign_keys = ON;")
        self._initialise_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def _initialise_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executescript(self._schema())
            conn.commit()

    def execute_query(self, query, params=(), fetchone=False, fetchall=False):
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if fetchone:
                    row = cursor.fetchone()
                    if row:
                        return dict(zip([desc[0] for desc in cursor.description], row))
                    return None
                if fetchall:
                    rows = cursor.fetchall()
                    return [dict(zip([desc[0] for desc in cursor.description], row)) for row in rows]  # ✅ Returns List[Dict]
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return None        

    def _schema(self):
        return """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS populations (
            id TEXT PRIMARY KEY,
            population_id TEXT, 
            household_id INTEGER,
            name TEXT,
            age INTEGER,
            gender TEXT,
            occupation TEXT,
            occupation_category INTEGER,
            relationship TEXT,
            FOREIGN KEY (population_id) REFERENCES metadata (population_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS metadata (
            population_id TEXT PRIMARY KEY, 
            location TEXT,
            model TEXT,
            temperature REAL,
            top_p REAL,
            top_k INTEGER,
            num_households INTEGER,
            execution_time REAL,
            prompt TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            include_stats BOOLEAN,
            include_guidance BOOLEAN,
            include_target BOOLEAN,
            compute_household_size BOOLEAN,
            use_microdata BOOLEAN,
            no_occupation BOOLEAN,
            no_household_composition BOOLEAN,
            include_avg_household_size BOOLEAN,
            hh_type_classifier TEXT,
            hh_size_classifier TEXT
        );

        CREATE TABLE IF NOT EXISTS estimation_metadata (
            run_id TEXT PRIMARY KEY,
            variable TEXT,
            model_name TEXT,
            n_trials INTEGER,
            prompt_template TEXT,
            schema_name TEXT,
            input_hash TEXT,
            run_timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS estimations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            variable TEXT,                  -- e.g., "population_size", "age_distribution"
            location TEXT,
            category TEXT,                  -- e.g., "Major", "Minor"
            subcategory TEXT,               -- e.g., "Aged 10–14 Female", "Unemployed"
            ground_truth REAL,
            trial_number INTEGER,
            prediction REAL,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES estimation_metadata(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY, 
            location TEXT,
            model TEXT,
            temperature REAL,
            top_p REAL,
            top_k INTEGER,
            execution_time REAL,
            prompt TEXT,
            prompt_name TEXT,
            num_households INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            include_stats BOOLEAN,
            include_guidance BOOLEAN,
            include_target BOOLEAN,
            compute_household_size BOOLEAN,
            use_microdata BOOLEAN,
            no_occupation BOOLEAN,
            no_household_composition BOOLEAN,
            include_avg_household_size BOOLEAN,
            hh_type_classifier TEXT,
            hh_size_classifier TEXT
        );

        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT,
            run_number INTEGER,
            population_id TEXT,
            execution_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
            FOREIGN KEY (population_id) REFERENCES metadata (population_id) ON DELETE CASCADE
        );
        """

