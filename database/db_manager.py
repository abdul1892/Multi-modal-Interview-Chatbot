import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))


class DatabaseManager:
    """
    MySQL-backed persistence layer.

    Schema:
      users             — registered accounts; passwords are bcrypt-hashed.
      interviews        — one row per session, owned by a user (FK with cascade).
      interview_rounds  — one row per question, with full prosody breakdown.

    Forward-compatible: _initialize_tables() migrates legacy schemas in place:
      - adds users.id FK to interviews if missing
      - adds determination/doubt/anxiety columns to interview_rounds if missing
    so an existing deployment upgrades on the next restart without losing data.
    """

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "ai_interviewer")

        self.connection = self._connect()
        if self.connection:
            self._initialize_tables()

    def _connect(self):
        try:
            temp_conn = mysql.connector.connect(
                host=self.host, user=self.user, password=self.password
            )
            if temp_conn.is_connected():
                cursor = temp_conn.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                cursor.close()
                temp_conn.close()

            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print(f"[DATABASE] Successfully connected to MySQL Database: '{self.database}'")
            return conn
        except Error as e:
            print(f"[ERROR] MySQL Connection Failed. Is your MySQL server running? Error: {e}")
            return None

    def _ensure_connection(self):
        """Reconnect if MySQL dropped us (wait_timeout, restart, etc.)."""
        try:
            if self.connection is None or not self.connection.is_connected():
                print("[DATABASE] Connection lost. Reconnecting...")
                self.connection = self._connect()
            else:
                self.connection.ping(reconnect=True, attempts=2, delay=1)
        except Error as e:
            print(f"[ERROR] Reconnection ping failed: {e}")
            self.connection = self._connect()
        return self.connection is not None

    def _column_exists(self, cursor, table, column):
        cursor.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (column,))
        return cursor.fetchone() is not None

    def _initialize_tables(self):
        if not self.connection:
            return
        try:
            cursor = self.connection.cursor()

            # ----- users -----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_email (email)
                )
            """)

            # ----- interviews -----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interviews (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    final_report TEXT,
                    radar_metrics VARCHAR(255),
                    INDEX idx_user (user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Legacy migration: add user_id if a pre-auth schema exists.
            if not self._column_exists(cursor, 'interviews', 'user_id'):
                print("[DATABASE] Migrating interviews: adding user_id...")
                cursor.execute("ALTER TABLE interviews ADD COLUMN user_id INT AFTER id")
                cursor.execute("ALTER TABLE interviews ADD INDEX idx_user (user_id)")
                cursor.execute("""
                    ALTER TABLE interviews
                    ADD CONSTRAINT fk_interviews_user
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                """)

            # ----- interview_rounds -----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interview_rounds (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    interview_id INT,
                    round_number INT,
                    topic VARCHAR(80),
                    question TEXT,
                    ideal_answer TEXT,
                    transcript TEXT,
                    tech_score FLOAT,
                    rubric_reason VARCHAR(500),
                    confidence_score FLOAT,
                    determination FLOAT DEFAULT 0,
                    doubt FLOAT DEFAULT 0,
                    anxiety FLOAT DEFAULT 0,
                    FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE
                )
            """)

            # Legacy migration: add any columns missing from older schemas.
            if not self._column_exists(cursor, 'interview_rounds', 'topic'):
                print("[DATABASE] Migrating interview_rounds: adding topic...")
                cursor.execute("ALTER TABLE interview_rounds ADD COLUMN topic VARCHAR(80) AFTER round_number")
            if not self._column_exists(cursor, 'interview_rounds', 'ideal_answer'):
                print("[DATABASE] Migrating interview_rounds: adding ideal_answer...")
                cursor.execute("ALTER TABLE interview_rounds ADD COLUMN ideal_answer TEXT AFTER question")
            if not self._column_exists(cursor, 'interview_rounds', 'rubric_reason'):
                print("[DATABASE] Migrating interview_rounds: adding rubric_reason...")
                cursor.execute("ALTER TABLE interview_rounds ADD COLUMN rubric_reason VARCHAR(500) AFTER tech_score")
            for col in ("determination", "doubt", "anxiety"):
                if not self._column_exists(cursor, 'interview_rounds', col):
                    print(f"[DATABASE] Migrating interview_rounds: adding {col}...")
                    cursor.execute(f"ALTER TABLE interview_rounds ADD COLUMN {col} FLOAT DEFAULT 0")

            self.connection.commit()
            cursor.close()
            print("[DATABASE] Relational tables verified and initialized.")
        except Error as e:
            print(f"[ERROR] Table Initialization Failed: {e}")

    # ---------------------------------------------------------------------
    # USERS — called by auth.py
    # ---------------------------------------------------------------------
    def create_user(self, username, email, password_hash):
        if not self._ensure_connection():
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash)
            )
            self.connection.commit()
            user_id = cursor.lastrowid
            cursor.close()
            print(f"[DATABASE] Created new user: '{username}' (id={user_id})")
            return user_id
        except Error as e:
            print(f"[ERROR] Failed to create user: {e}")
            return None

    def get_user_by_id(self, user_id):
        if not self._ensure_connection():
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE id = %s",
                (user_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            return row
        except Error as e:
            print(f"[ERROR] get_user_by_id failed: {e}")
            return None

    def get_user_by_username(self, username):
        if not self._ensure_connection():
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE username = %s",
                (username,)
            )
            row = cursor.fetchone()
            cursor.close()
            return row
        except Error as e:
            print(f"[ERROR] get_user_by_username failed: {e}")
            return None

    def get_user_by_email(self, email):
        if not self._ensure_connection():
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE email = %s",
                (email,)
            )
            row = cursor.fetchone()
            cursor.close()
            return row
        except Error as e:
            print(f"[ERROR] get_user_by_email failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # INTERVIEWS
    # ---------------------------------------------------------------------
    def create_interview_session(self, user_id):
        """Creates a new interview session owned by user_id."""
        if not self._ensure_connection() or not user_id:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO interviews (user_id, final_report) VALUES (%s, NULL)",
                (user_id,)
            )
            self.connection.commit()
            interview_id = cursor.lastrowid
            cursor.close()
            print(f"[DATABASE] Created interview {interview_id} for user_id={user_id}")
            return interview_id
        except Error as e:
            print(f"[ERROR] Failed to create session: {e}")
            return None

    def log_round(self, interview_id, round_number, question, transcript,
                  tech_score, confidence_score,
                  determination=0.0, doubt=0.0, anxiety=0.0,
                  topic=None, ideal_answer=None, rubric_reason=None):
        """Persist a single round with all the data the report PDF needs."""
        if not self._ensure_connection() or not interview_id:
            return
        try:
            cursor = self.connection.cursor()
            sql = """
                INSERT INTO interview_rounds
                (interview_id, round_number, topic, question, ideal_answer,
                 transcript, tech_score, rubric_reason, confidence_score,
                 determination, doubt, anxiety)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (interview_id, round_number, topic, question, ideal_answer,
                      transcript, tech_score, rubric_reason, confidence_score,
                      determination, doubt, anxiety)
            cursor.execute(sql, values)
            self.connection.commit()
            cursor.close()
            print(f"[DATABASE] Logged Round {round_number} ({topic}) for Session {interview_id}.")
        except Error as e:
            print(f"[ERROR] Failed to log round: {e}")

    def complete_interview(self, interview_id, final_report, radar_metrics):
        if not self._ensure_connection() or not interview_id:
            return
        try:
            cursor = self.connection.cursor()
            metrics_str = ",".join(map(str, radar_metrics))
            sql = "UPDATE interviews SET final_report = %s, radar_metrics = %s WHERE id = %s"
            cursor.execute(sql, (final_report, metrics_str, interview_id))
            self.connection.commit()
            cursor.close()
            print(f"[DATABASE] Final evaluation saved for Session {interview_id}.")
        except Error as e:
            print(f"[ERROR] Failed to complete interview: {e}")

    def get_interview(self, interview_id):
        """Full interview row — used by /download-report/<id>."""
        if not self._ensure_connection() or not interview_id:
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, user_id, session_date, final_report, radar_metrics
                FROM interviews WHERE id = %s
            """, (interview_id,))
            row = cursor.fetchone()
            cursor.close()
            return row
        except Error as e:
            print(f"[ERROR] get_interview failed: {e}")
            return None

    def get_interview_owner(self, interview_id):
        """Lightweight ownership check."""
        if not self._ensure_connection() or not interview_id:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT user_id FROM interviews WHERE id = %s", (interview_id,))
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else None
        except Error as e:
            print(f"[ERROR] get_interview_owner failed: {e}")
            return None

    def get_interview_rounds(self, interview_id):
        """All rounds for an interview, ordered. Used to reconstruct archived PDFs."""
        if not self._ensure_connection() or not interview_id:
            return []
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT round_number, topic, question, ideal_answer, transcript,
                       tech_score, rubric_reason, confidence_score,
                       determination, doubt, anxiety
                FROM interview_rounds
                WHERE interview_id = %s
                ORDER BY round_number
            """, (interview_id,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            print(f"[ERROR] get_interview_rounds failed: {e}")
            return []

    def delete_interview_if_empty(self, interview_id):
        """
        Delete an interview row only if it has zero rounds. Used when a user
        starts an interview, never answers a question, then disconnects /
        signs out — we don't want orphan rows cluttering their history.

        Safe to call on any interview_id: if there are rounds, this no-ops.
        """
        if not self._ensure_connection() or not interview_id:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM interview_rounds WHERE interview_id = %s",
                (interview_id,)
            )
            round_count = cursor.fetchone()[0]
            if round_count == 0:
                cursor.execute("DELETE FROM interviews WHERE id = %s", (interview_id,))
                self.connection.commit()
                print(f"[DATABASE] Deleted empty interview {interview_id} (0 rounds).")
                cursor.close()
                return True
            cursor.close()
            return False
        except Error as e:
            print(f"[ERROR] delete_interview_if_empty failed: {e}")
            return False

    def get_user_trend_data(self, user_id, up_to_interview_id=None):
        """
        Per-interview aggregates for every COMPLETED interview owned by the user,
        ordered OLDEST-first. Powers the trend chart in both the PDF report and
        the My Interviews page.

        Prosody fields (determination/doubt/anxiety) are resume-independent, so
        they're safe to compare across sessions even when the skills differ.

        If up_to_interview_id is given, only interviews on or before that
        interview's session_date are included — so an archived report shows the
        trend as it stood at that time, not sessions that happened afterwards.
        """
        if not self._ensure_connection() or not user_id:
            return []
        try:
            cursor = self.connection.cursor(dictionary=True)
            params = [user_id]
            date_clause = ""
            if up_to_interview_id:
                date_clause = (
                    "AND i.session_date <= "
                    "(SELECT session_date FROM interviews WHERE id = %s) "
                )
                params.append(up_to_interview_id)

            cursor.execute(f"""
                SELECT
                    i.id,
                    i.session_date,
                    COUNT(r.id) AS round_count,
                    COALESCE(AVG(r.tech_score), 0)       AS avg_tech,
                    COALESCE(AVG(r.confidence_score), 0) AS avg_conf,
                    COALESCE(AVG(r.determination), 0)    AS avg_determination,
                    COALESCE(AVG(r.doubt), 0)            AS avg_doubt,
                    COALESCE(AVG(r.anxiety), 0)          AS avg_anxiety
                FROM interviews i
                LEFT JOIN interview_rounds r ON r.interview_id = i.id
                WHERE i.user_id = %s
                  AND i.final_report IS NOT NULL
                  {date_clause}
                GROUP BY i.id
                HAVING COUNT(r.id) > 0
                ORDER BY i.session_date ASC, i.id ASC
            """, tuple(params))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            print(f"[ERROR] get_user_trend_data failed: {e}")
            return []

    def get_user_interviews(self, user_id):
        """
        All interview sessions owned by this user, newest first. Includes
        per-row aggregates so the listing page renders with one query.
        """
        if not self._ensure_connection() or not user_id:
            return []
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT
                    i.id,
                    i.session_date,
                    i.radar_metrics,
                    (i.final_report IS NOT NULL) AS is_complete,
                    COUNT(r.id) AS round_count,
                    COALESCE(AVG(r.tech_score), 0) AS avg_tech_score,
                    COALESCE(AVG(r.confidence_score), 0) AS avg_conf_score
                FROM interviews i
                LEFT JOIN interview_rounds r ON r.interview_id = i.id
                WHERE i.user_id = %s
                GROUP BY i.id
                HAVING COUNT(r.id) > 0
                ORDER BY i.session_date DESC
            """, (user_id,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            print(f"[ERROR] get_user_interviews failed: {e}")
            return []
