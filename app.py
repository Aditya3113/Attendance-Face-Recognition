import sqlite3
import json
import base64
import io
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_file,
    session,
)

import numpy as np
from PIL import Image
from deepface import DeepFace
import pandas as pd



DB_PATH = "faces.db"

LOCATIONS = [
    "College Main Door",
    "Gym",
    "Transit House",
    "Library",
    "Hostel",
    "Study Room",
]


ADMIN_USERNAME = "iiitu"
ADMIN_PASSWORD = "neha1234"



class Database:
    """Handles all SQLite-related operations."""

    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist, and ensure schema is up to date."""
        conn = sqlite3.connect(self.path)
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll_no TEXT NOT NULL UNIQUE,
                branch TEXT,
                semester TEXT,
                face_embedding TEXT NOT NULL
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                direction TEXT NOT NULL,     -- 'entry' or 'exit'
                timestamp TEXT NOT NULL,     -- ISO string
                location TEXT NOT NULL DEFAULT 'College Main Door',
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )

        
        c.execute("PRAGMA table_info(movements)")
        cols = [row[1] for row in c.fetchall()]
        if "location" not in cols:
            c.execute(
                "ALTER TABLE movements ADD COLUMN location TEXT NOT NULL DEFAULT 'College Main Door'"
            )

        conn.commit()
        conn.close()

    def get_connection(self):
        """Return a new DB connection."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn



class FaceEncoder:
    """Encapsulates face embedding logic using DeepFace."""

    def __init__(self, model_name: str = "Facenet"):
        self.model_name = model_name

    def get_embedding_from_data_url(self, data_url: str):

        if not data_url or "," not in data_url:
            return None, "No image data received."

        try:
            _, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        except Exception as e:
            return None, f"Failed to decode image: {e}"

        try:
            image = Image.open(io.BytesIO(img_bytes))
            image = image.convert("RGB")
            img_np = np.array(image)
        except Exception as e:
            return None, f"Failed to load image: {e}"

        try:
            reps = DeepFace.represent(
                img_path=img_np,
                model_name=self.model_name,
                enforce_detection=True,
            )
            if isinstance(reps, list) and len(reps) > 0:
                emb = np.array(reps[0]["embedding"], dtype="float32")
                return emb, None
            else:
                return None, "No face embedding generated. Try again."
        except Exception as e:
            return None, f"Face analysis error: {e}"

    @staticmethod
    def cosine_similarity(a, b) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)



class MovementManager:

    def __init__(self, db: Database, min_gap_seconds: int = 60):
        self.db = db
        self.min_gap_seconds = min_gap_seconds

    def log_movement(self, student_id: int, location: str):
    
        conn = self.db.get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, direction, timestamp
            FROM movements
            WHERE student_id = ? AND location = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (student_id, location),
        )
        row = cur.fetchone()

        now_dt = datetime.now()
        now_str = now_dt.isoformat(sep=" ", timespec="seconds")

        if row:
            last_direction = row["direction"]
            last_ts_str = row["timestamp"]

            try:
                last_dt = datetime.fromisoformat(last_ts_str)
                diff = (now_dt - last_dt).total_seconds()
            except Exception:
                diff = self.min_gap_seconds + 1

            if diff < self.min_gap_seconds:
                conn.close()
                return last_direction, last_ts_str, False

            new_direction = "exit" if last_direction == "entry" else "entry"

        else:
            if location in ("College Main Door", "Hostel"):
                new_direction = "exit"
            else:
                new_direction = "entry"

        cur.execute(
            """
            INSERT INTO movements (student_id, direction, timestamp, location)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, new_direction, now_str, location),
        )
        conn.commit()
        conn.close()

        return new_direction, now_str, True

    def get_live_counts(self):

        conn = self.db.get_connection()
        cur = conn.cursor()

        stats = []

        for loc in LOCATIONS:
            cur.execute(
                """
                SELECT m.student_id, m.direction, m.timestamp
                FROM movements m
                JOIN (
                    SELECT student_id, MAX(timestamp) AS max_ts
                    FROM movements
                    WHERE location = ?
                    GROUP BY student_id
                ) latest
                ON m.student_id = latest.student_id
                AND m.timestamp = latest.max_ts
                WHERE m.location = ?
                """,
                (loc, loc),
            )
            rows = cur.fetchall()

            count = 0
            if loc in ("College Main Door", "Hostel"):
                for r in rows:
                    if r["direction"] == "exit":
                        count += 1
                mode = "outside"
            else:
                for r in rows:
                    if r["direction"] == "entry":
                        count += 1
                mode = "inside"

            stats.append({
                "location": loc,
                "count": count,
                "mode": mode,
            })

        conn.close()
        return stats


class RecognitionService:
    
    def __init__(self, db: Database, encoder: FaceEncoder, threshold: float = 0.65):
        self.db = db
        self.encoder = encoder
        self.threshold = threshold

    def _load_all_students(self):
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, roll_no, branch, semester, face_embedding FROM students"
        )
        rows = cur.fetchall()
        conn.close()

        students = []
        for row in rows:
            try:
                emb_list = json.loads(row["face_embedding"])
                emb = np.array(emb_list, dtype="float32")
            except Exception:
                emb = None

            students.append({
                "id": row["id"],
                "name": row["name"],
                "roll_no": row["roll_no"],
                "branch": row["branch"],
                "semester": row["semester"],
                "embedding": emb,
            })

        return students

    def recognize_from_data_url(self, data_url: str):
        unknown_embedding, error = self.encoder.get_embedding_from_data_url(data_url)
        if error:
            return None, None, error

        students = self._load_all_students()
        if not students:
            return None, None, "No registered faces in database. Please register someone first."

        best_similarity = -1.0
        best_student = None

        for s in students:
            if s["embedding"] is None:
                continue
            sim = self.encoder.cosine_similarity(unknown_embedding, s["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_student = {
                    "id": s["id"],
                    "name": s["name"],
                    "roll_no": s["roll_no"],
                    "branch": s["branch"],
                    "semester": s["semester"],
                }

        return best_student, best_similarity, None




app = Flask(__name__)
app.config["SECRET_KEY"] = "change_this_to_a_random_secret_key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  


db = Database(DB_PATH)
face_encoder = FaceEncoder(model_name="Facenet")
movement_manager = MovementManager(db, min_gap_seconds=60)
recognition_service = RecognitionService(db, face_encoder, threshold=0.65)



def require_admin():
    return bool(session.get("admin_logged_in"))



@app.route("/")
def index():
    return redirect(url_for("admin_login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if not require_admin():
        flash("Admin login required.")
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        branch = request.form.get("branch", "").strip()
        semester = request.form.get("semester", "").strip()
        captured_image = request.form.get("captured_image", "")

        if not name or not roll_no:
            flash("Name and Roll Number are required.")
            return redirect(url_for("register"))

        emb, error = face_encoder.get_embedding_from_data_url(captured_image)
        if error or emb is None:
            flash(error or "No face found. Please try again.")
            return redirect(url_for("register"))

        embedding_json = json.dumps(emb.tolist())

        try:
            conn = db.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO students (name, roll_no, branch, semester, face_embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, roll_no, branch, semester, embedding_json),
            )
            conn.commit()
            conn.close()
            flash("Student registered successfully!")
        except sqlite3.IntegrityError:
            flash("Roll number already exists. Use a unique roll number.")
        except Exception as e:
            flash(f"Database error: {e}")

        return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/recognize")
def recognize():
    if not require_admin():
        flash("Admin login required.")
        return redirect(url_for("admin_login"))

    default_location = LOCATIONS[0]
    return render_template(
        "recognize.html",
        locations=LOCATIONS,
        default_location=default_location,
    )


@app.route("/api/recognize_frame", methods=["POST"])
def api_recognize_frame():
    if not require_admin():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    data = request.get_json(silent=True)
    if not data or "captured_image" not in data:
        return jsonify({"status": "error", "message": "No image data in request"}), 400

    captured_image = data["captured_image"]
    location = data.get("location") or LOCATIONS[0]

    if location not in LOCATIONS:
        return jsonify({"status": "error", "message": "Invalid location"}), 400

    student, similarity, error = recognition_service.recognize_from_data_url(
        captured_image
    )

    if error:
        return jsonify({"status": "error", "message": error}), 200

    if not student or similarity is None:
        return jsonify({
            "status": "nomatch",
            "message": "No matching person found in database.",
            "similarity": similarity if similarity is not None else 0.0,
        }), 200

    if similarity >= recognition_service.threshold:
        direction, ts_str, created = movement_manager.log_movement(
            student["id"], location
        )
        return jsonify({
            "status": "ok",
            "student": student,
            "similarity": similarity,
            "direction": direction,
            "timestamp": ts_str,
            "location": location,
            "new_log": created,
        }), 200
    else:
        return jsonify({
            "status": "nomatch",
            "message": "No matching person found in database.",
            "similarity": similarity,
        }), 200


@app.route("/logs")
def logs():
     
    if not require_admin():
        flash("Admin login required.")
        return redirect(url_for("admin_login"))

    location = request.args.get("location")
    if not location or location not in LOCATIONS:
        location = LOCATIONS[0]

    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.id,
            m.direction,
            m.timestamp,
            m.location,
            s.name,
            s.roll_no,
            s.branch,
            s.semester
        FROM movements m
        JOIN students s ON m.student_id = s.id
        WHERE m.location = ?
        ORDER BY m.timestamp DESC
        """,
        (location,),
    )
    records = cur.fetchall()
    conn.close()

    return render_template(
        "logs.html",
        records=records,
        locations=LOCATIONS,
        current_location=location,
    )


@app.route("/logs/export")
def export_logs():
     
    if not require_admin():
        flash("Admin login required.")
        return redirect(url_for("admin_login"))

    location = request.args.get("location")
    if not location or location not in LOCATIONS:
        location = LOCATIONS[0]

    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.timestamp,
            m.direction,
            m.location,
            s.name,
            s.roll_no,
            s.branch,
            s.semester
        FROM movements m
        JOIN students s ON m.student_id = s.id
        WHERE m.location = ?
        ORDER BY m.timestamp DESC
        """,
        (location,),
    )
    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "Timestamp": r["timestamp"],
            "Direction": r["direction"].upper(),
            "Location": r["location"],
            "Name": r["name"],
            "Roll No": r["roll_no"],
            "Branch": r["branch"],
            "Semester": r["semester"],
        })

    df = pd.DataFrame(data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Logs")
    output.seek(0)

    safe_loc = location.replace(" ", "_")
    filename = f"logs_{safe_loc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/dashboard")
def dashboard():
     
    stats = movement_manager.get_live_counts()
    return render_template(
        "dashboard.html",
        stats=stats,
        locations=LOCATIONS,
    )


@app.route("/analytics")
def analytics():
     
    if not require_admin():
        flash("Admin login required.")
        return redirect(url_for("admin_login"))

    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.now().date().isoformat()

    conn = db.get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            location,
            COUNT(*) AS total_movements,
            SUM(CASE WHEN direction = 'entry' THEN 1 ELSE 0 END) AS entries,
            SUM(CASE WHEN direction = 'exit' THEN 1 ELSE 0 END) AS exits,
            COUNT(DISTINCT student_id) AS unique_students
        FROM movements
        WHERE DATE(timestamp) = ?
        GROUP BY location
        ORDER BY location
        """,
        (date_str,),
    )
    per_location_rows = cur.fetchall()
    per_location = [dict(row) for row in per_location_rows]

    cur.execute(
        """
        SELECT
            COUNT(*) AS total_movements,
            SUM(CASE WHEN direction = 'entry' THEN 1 ELSE 0 END) AS entries,
            SUM(CASE WHEN direction = 'exit' THEN 1 ELSE 0 END) AS exits,
            COUNT(DISTINCT student_id) AS unique_students
        FROM movements
        WHERE DATE(timestamp) = ?
        """,
        (date_str,),
    )
    overall_row = cur.fetchone()
    overall = dict(overall_row) if overall_row else None

    conn.close()

    return render_template(
        "analytics.html",
        date_str=date_str,
        per_location=per_location,
        overall=overall,
    )



@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["admin_logged_in"] = True
            flash("Logged in successfully as admin.")
            return redirect(url_for("register"))
        else:
            flash("Invalid username or password.")

    return render_template("admin_login.html")


@app.route("/guest")
def guest_login():
     
    session.clear()
    session["guest"] = True
    flash("You are viewing as guest. Only the dashboard is available.")
    return redirect(url_for("dashboard"))


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("admin_login"))


@app.route("/student_status", methods=["GET", "POST"])
def student_status():
    if not require_admin():
        flash("Please login as admin to access student status.")
        return redirect(url_for("admin_login"))

    result = None
    error = None

    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()

        if not roll_no:
            error = "Please enter a roll number."
        else:
            conn = db.get_connection()
            cur = conn.cursor()

            cur.execute(
                """
                SELECT id, name, roll_no, branch, semester
                FROM students
                WHERE roll_no = ?
                """,
                (roll_no,),
            )
            student = cur.fetchone()

            if not student:
                error = f"No student found with roll number {roll_no}."
                conn.close()
            else:
                student_id = student["id"]

                cur.execute(
                    """
                    SELECT direction, timestamp, location
                    FROM movements
                    WHERE student_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (student_id,),
                )
                movement = cur.fetchone()
                conn.close()

                if not movement:
                    result = {
                        "student": student,
                        "has_movement": False,
                        "status_text": "No entry/exit records yet.",
                    }
                else:
                    direction = movement["direction"]
                    ts = movement["timestamp"]
                    location = movement["location"]

                   
                    if location in ("College Main Door", "Hostel"):
                        if direction == "exit":
                            status_text = f"Outside (last seen exiting {location})."
                        else:
                            status_text = f"Inside {location} (last seen entering)."
                    else:
                        if direction == "entry":
                            status_text = f"Inside {location} (last seen entering)."
                        else:
                            status_text = f"Outside {location} (last seen exiting)."

                    result = {
                        "student": student,
                        "has_movement": True,
                        "direction": direction,
                        "timestamp": ts,
                        "location": location,
                        "status_text": status_text,
                    }

    return render_template(
        "student_status.html",
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)