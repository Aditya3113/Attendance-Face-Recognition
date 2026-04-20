# Attendance Management using Face Recognition (DeepFace + Flask)

This project is a **Flask web application** that records **student attendance / movement (entry/exit)** using **face recognition**. It supports **multiple locations** (e.g., Main Door, Library, Hostel), provides **live counts**, **logs export to Excel**, **daily analytics**, and a **student status lookup** by roll number.

---

## Tech Stack

- **Backend**: Python, Flask
- **Face recognition**: `deepface` (model: **Facenet**) for face embeddings
- **Database**: SQLite (`faces.db`)
- **Frontend**: HTML + inline CSS/JS (camera capture using `getUserMedia`)
- **Analytics chart**: local Chart.js (`static/chart.umd.min.js`)
- **Export**: `pandas` + `openpyxl` (Excel `.xlsx`)

---

## Project Structure

- `app.py`: Main Flask app (routes, DB logic, face embedding + recognition, movement logging)
- `templates/`: All HTML pages rendered by Flask
  - `admin_login.html`
  - `register.html`
  - `recognize.html`
  - `logs.html`
  - `dashboard.html`
  - `analytics.html`
  - `student_status.html`
- `static/`
  - `chart.umd.min.js`: Chart.js used by `analytics.html`
- `requirements.txt`: Python dependencies
- `faces.db`: SQLite database file (auto-created at runtime)

---

## Setup & Run

### 1) Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the server

```bash
python app.py
```

The app starts with Flask debug mode and listens on `0.0.0.0`.

Open:
- `http://127.0.0.1:5000/`

---

## Login (Admin / Guest)

### Admin login

Admin access is required for **registration**, **recognition**, **logs**, **analytics**, and **student status**.

The credentials are hardcoded in `app.py`:
- **Username**: `iiitu`
- **Password**: `neha1234`

### Guest access

Guest mode only allows viewing the **Dashboard** (live counts).  
Use the "Continue as Guest" button on the login page.

---

## How Face Recognition Works (DeepFace / Facenet)

This project does **not** store raw face images in the database. It stores a **face embedding** (a numeric feature vector) produced by DeepFace.

### Registration flow (embedding creation)

1. `register.html` captures a webcam frame and submits it as a **base64 data URL** (`captured_image`).
2. `FaceEncoder.get_embedding_from_data_url()` in `app.py`:
   - Decodes base64 → image bytes
   - Loads image using PIL → converts to RGB → NumPy array
   - Calls:
     - `DeepFace.represent(img_path=img_np, model_name="Facenet", enforce_detection=True)`
3. The resulting embedding is converted to JSON and stored in SQLite (`students.face_embedding`).

**Important**: `enforce_detection=True` means registration/recognition will fail if DeepFace cannot detect a face in the frame.

### Recognition flow (matching)

1. `recognize.html` continuously captures frames (every ~1.5 seconds) and sends them to:
   - `POST /api/recognize_frame` with JSON:
     - `captured_image`: base64 JPEG data URL
     - `location`: selected location name
2. The server creates an embedding for the incoming frame (same method as registration).
3. It loads all registered students and compares embeddings using **cosine similarity**:

\[
\text{cosine\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}
\]

4. The best match is accepted if similarity \(\ge\) **threshold**.
   - Threshold is set in `RecognitionService(threshold=0.65)`.

### Why embeddings + cosine similarity?

- Embeddings are a compact representation of facial features.
- Cosine similarity is commonly used to compare embedding vectors.
- This approach is fast enough for small/medium student lists and works offline (no external APIs).

---

## Attendance / Movement Logic (Entry vs Exit)

Movements are stored per student **per location**. For a given student+location, the system alternates:
- `entry` → `exit` → `entry` → ...

To reduce duplicate logs caused by repeated frames:
- A **minimum time gap** of **60 seconds** is enforced (`min_gap_seconds=60`).
  - If the last movement at the same location is within 60 seconds, a new log is **not created**.

### Live counts meaning (Dashboard)

In `dashboard.html`:
- For **College Main Door** and **Hostel**: the count shows how many students are currently **outside** (last movement there is `exit`).
- For all other locations: the count shows how many students are currently **inside** (last movement there is `entry`).

---

## Database Schema (SQLite)

The database file is `faces.db` (created automatically).

### `students` table

- `id` (PK)
- `name`
- `roll_no` (unique)
- `branch`
- `semester`
- `face_embedding` (JSON string list of floats)

### `movements` table

- `id` (PK)
- `student_id` (FK → `students.id`)
- `direction` (`entry` or `exit`)
- `timestamp` (ISO-like string)
- `location` (one of the configured location names)

---

## Routes / Pages

### Main routes

- `/` → redirects to admin login
- `/admin/login` → Admin login page
- `/guest` → Guest session → redirects to dashboard
- `/admin/logout` → Clears session

### Core attendance routes (admin required)

- `/register` → Register a student and save face embedding
- `/recognize` → Live recognition page (multi-location)
- `/api/recognize_frame` (POST) → Receives a webcam frame, returns recognition + movement logging result
- `/logs` → View movement logs (filter by location)
- `/logs/export` → Export logs to Excel for the selected location
- `/analytics` → Daily analytics page (per-location totals + pie chart)
- `/student_status` → Look up a student’s last known status by roll number

---

## HTML Templates Explained

### `templates/admin_login.html`

**Purpose**: Entry point for the application.

- **Admin login form**: posts username/password to `/admin/login`
- **Flash messages area**: shows login errors/success messages
- **Guest access**: link to `/guest` which allows viewing **only** the Dashboard
- **Navigation**: includes a link to `Dashboard` even before admin login

### `templates/register.html`

**Purpose**: Register a new student and store their face embedding.

- **Student details form**:
  - `name`, `roll_no` (required)
  - `branch`, `semester` (optional)
  - hidden `captured_image` (base64 data URL)
- **Camera panel**:
  - Uses `navigator.mediaDevices.getUserMedia({ video: true })`
  - Captures a frame to `<canvas>`, compresses to JPEG (`toDataURL("image/jpeg", 0.7)`)
  - Shows preview and fills the hidden input
- **Submit protection**:
  - `checkCaptured()` prevents saving unless a photo is captured

### `templates/recognize.html`

**Purpose**: Live recognition + automatic entry/exit logging with location selection.

- **Location dropdown**: sends selected location with each recognition request
- **Continuous recognition loop**:
  - Every 1.5 seconds it captures a resized webcam frame
  - Calls `POST /api/recognize_frame` with JSON `{captured_image, location}`
- **Pause/Resume button**: stops/starts the recognition loop on the client
- **Result panel**:
  - On success: shows student details + `ENTRY/EXIT` + time + location
  - On no match: shows best similarity score
  - On error: shows server-side errors (e.g., no face detected)

### `templates/logs.html`

**Purpose**: View movement logs and export them.

- **Location filter**:
  - Dropdown triggers GET `/logs?location=...`
- **Logs table**:
  - Lists movements for the selected location (latest first)
  - Direction is color-coded (`ENTRY` green, `EXIT` red)
- **Export to Excel**:
  - Link to `/logs/export?location=...`
  - Downloads `.xlsx` generated via `pandas` + `openpyxl`

### `templates/dashboard.html`

**Purpose**: Live overview of crowd distribution / counts across all locations.

- Shows one card per location with:
  - location name
  - current count
  - badge meaning (“Currently inside” or “Currently outside”)
- Auto-refresh:
  - Reloads the page every 10 seconds to update counts

### `templates/analytics.html`

**Purpose**: Daily analytics summary + per-location breakdown + chart.

- **Date filter**: GET `/analytics?date=YYYY-MM-DD`
- **Summary metrics**:
  - total movements, total entries, total exits, unique students
- **Per-location table**:
  - totals and unique counts grouped by location
- **Pie chart**:
  - Uses local Chart.js from `static/chart.umd.min.js`
  - Renders “Entries per Location” if entries > 0

### `templates/student_status.html`

**Purpose**: Query a student’s last known status by roll number.

- Form posts roll number to `/student_status`
- Shows:
  - student metadata (branch/semester)
  - last movement direction, location, timestamp
  - computed human-readable status:
    - for Main Door/Hostel: “Outside” if last movement was `exit`
    - for other locations: “Inside” if last movement was `entry`

---

## Configurable Values (in `app.py`)

- **Database file**: `DB_PATH = "faces.db"`
- **Locations list**: `LOCATIONS = [...]`
- **Recognition model**: `FaceEncoder(model_name="Facenet")`
- **Similarity threshold**: `threshold = 0.65`
- **Duplicate prevention window**: `min_gap_seconds = 60`

---

## Notes / Limitations

- **Hardcoded secrets**: `SECRET_KEY`, admin username/password are hardcoded for a college project/demo. For real deployment, move these to environment variables.
- **Performance**: Recognition compares the input embedding against *all* registered students in Python. This is fine for small datasets; for larger datasets, you’d typically use an embedding index (FAISS, Annoy, etc.).
- **Camera permissions**: Browser must allow webcam access. Use HTTPS in production (browsers restrict camera access on insecure contexts).

---

## Demo Flow (Quick Start)

1. Go to `/admin/login` and login as admin.
2. Open `/register` and register a student (capture photo + save).
3. Open `/recognize`, pick a location, stand in front of camera.
4. Check `/logs` for recorded movements and export Excel if needed.
5. Use `/dashboard` for live counts.
6. Use `/analytics` to see daily summary and entries-per-location chart.
7. Use `/student_status` to check a student’s last known state by roll number.

