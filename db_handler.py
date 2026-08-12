import sqlite3
import json
import bcrypt


DB_FILE = "database.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """Initialize database tables and safely migrate existing databases."""

    with get_connection() as conn:
        cursor = conn.cursor()

        # ----------------------------------------------------
        # Users table
        # ----------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ----------------------------------------------------
        # Safely add profile columns to an existing database
        # ----------------------------------------------------
        existing_columns = {
            row["name"]
            for row in cursor.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "age" not in existing_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN age INTEGER"
            )

        if "fitness_goal" not in existing_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN fitness_goal TEXT"
            )

        if "experience_level" not in existing_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN experience_level TEXT"
            )

        # ----------------------------------------------------
        # Exercise history table
        # ----------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercise_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise_name TEXT NOT NULL,
                movement_score REAL,
                control_score REAL,
                reps INTEGER,
                good_reps INTEGER,
                main_issue TEXT,
                unstable_events INTEGER,
                metrics_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.commit()


# Initialize database when this module is imported
init_db()


# ============================================================
# AUTHENTICATION
# ============================================================

def register_user(
    username,
    email,
    password,
    age,
    fitness_goal,
    experience_level
):
    """Register a new user with a securely hashed password."""

    username = (username or "").strip()
    email = (email or "").strip().lower()

    if not username:
        return False, "Username is required."

    if not email:
        return False, "Email is required."

    if not password:
        return False, "Password is required."

    if age is None:
        return False, "Age is required."

    # Hash password using bcrypt
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    age,
                    fitness_goal,
                    experience_level
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    password_hash,
                    int(age),
                    fitness_goal,
                    experience_level
                )
            )

            conn.commit()

        return True, "Account created successfully! Please sign in."

    except sqlite3.IntegrityError as e:
        error_message = str(e).lower()

        if "username" in error_message:
            return False, "That username is already taken."

        if "email" in error_message:
            return False, "An account with that email already exists."

        return False, "Username or email already exists."

    except Exception as e:
        return False, f"Registration error: {e}"


def authenticate_user(username, password):
    """Verify username and password using bcrypt."""

    if not username or not password:
        return False, "Please enter username and password."

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM users
                WHERE username = ?
                """,
                (username.strip(),)
            )

            user = cursor.fetchone()

        if user is None:
            return False, "Invalid username or password."

        stored_hash = user["password_hash"]

        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        ):
            return True, dict(user)

        return False, "Invalid username or password."

    except Exception as e:
        return False, f"Authentication error: {e}"


# ============================================================
# USER PROFILE
# ============================================================

def get_user(user_id):
    """Retrieve a user's profile by ID."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                username,
                email,
                age,
                fitness_goal,
                experience_level,
                created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        )

        user = cursor.fetchone()

    return dict(user) if user else None


def update_user_profile(
    user_id,
    age=None,
    fitness_goal=None,
    experience_level=None
):
    """Update optional user profile information."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET
                age = ?,
                fitness_goal = ?,
                experience_level = ?
            WHERE id = ?
            """,
            (
                age,
                fitness_goal,
                experience_level,
                user_id
            )
        )

        conn.commit()

    return True


# ============================================================
# EXERCISE LOGS
# ============================================================

def save_exercise_log(user_id, session_data):
    """Save a completed exercise session."""

    if not user_id:
        return False

    session_data = session_data or {}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO exercise_logs (
                    user_id,
                    exercise_name,
                    movement_score,
                    control_score,
                    reps,
                    good_reps,
                    main_issue,
                    unstable_events,
                    metrics_data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_data.get(
                        "exercise_name",
                        "Unknown"
                    ),
                    session_data.get(
                        "movement_score"
                    ),
                    session_data.get(
                        "control_score"
                    ),
                    session_data.get(
                        "reps",
                        0
                    ),
                    session_data.get(
                        "good_reps",
                        0
                    ),
                    session_data.get(
                        "main_issue",
                        ""
                    ),
                    session_data.get(
                        "unstable_events",
                        0
                    ),
                    json.dumps(
                        session_data.get(
                            "metric_scores",
                            {}
                        )
                    )
                )
            )

            conn.commit()

        return True

    except Exception as e:
        print(f"Error saving exercise log: {e}")
        return False


def get_user_sessions(user_id):
    """Retrieve all exercise sessions for a user."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM exercise_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

    sessions = []

    for row in rows:
        data = dict(row)

        # Safely decode metrics JSON
        try:
            data["metric_scores"] = (
                json.loads(data["metrics_data"])
                if data["metrics_data"]
                else {}
            )
        except (json.JSONDecodeError, TypeError):
            data["metric_scores"] = {}

        # Extract date from timestamp
        data["date"] = (
            data["timestamp"].split(" ")[0]
            if data["timestamp"]
            else ""
        )

        sessions.append(data)

    return sessions


def clear_user_history(user_id):
    """Delete all exercise history for a user."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM exercise_logs
            WHERE user_id = ?
            """,
            (user_id,)
        )

        conn.commit()

def get_user_profile(user_id):
    """Get the profile information for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, email, age, fitness_goal, experience_level
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        )

        user = cursor.fetchone()

        if user:
            return dict(user)

        return None


def update_user_profile(
    user_id,
    email,
    age,
    fitness_goal,
    experience_level
):
    """Update editable user profile information."""

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE users
                SET email = ?,
                    age = ?,
                    fitness_goal = ?,
                    experience_level = ?
                WHERE id = ?
                """,
                (
                    email.strip().lower(),
                    age,
                    fitness_goal,
                    experience_level,
                    user_id
                )
            )

            conn.commit()

        return True, "Profile updated successfully."

    except sqlite3.IntegrityError:
        return False, "That email is already associated with another account."

    except Exception as e:
        return False, f"Could not update profile: {e}"

    return True