# OvaScan AI - Collaborator Setup Guide

This guide explains how to set up the OvaScan AI project on a new laptop.

## 📋 Prerequisites
Ensure you have the following installed:
1.  **Node.js** (v18 or higher)
2.  **Python** (v3.10 or higher)
3.  **Git**

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ovariantumor_and_cancer
```

### 2. Backend Setup (Python)
The backend handles AI Inference and Firebase connections.

1.  **Navigate to backend:**
    ```bash
    cd backend
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **🔑 CRITICAL: Add Firebase Credentials**
    *   This project requires a `serviceAccountKey.json` file to connect to the database.
    *   **This file is ignored by Git for security.**
    *   **Action:** Ask the project owner to securely send you their `serviceAccountKey.json`.
    *   **Place it here:** `backend/serviceAccountKey.json`

5.  **Start the Backend Server:**
    ```bash
    python main.py
    ```
    ✅ Server running on: `http://localhost:8002`

---

### 3. Frontend Setup (React/Vite)
Open a **new terminal** window (do not close the backend terminal).

1.  **Navigate to project root:**
    ```bash
    cd ovariantumor_and_cancer # If not already there
    ```

2.  **Install Node Modules:**
    ```bash
    npm install
    ```

3.  **Start the Frontend:**
    ```bash
    npm run dev
    ```
    ✅ App accessible at: `http://localhost:3000`

---

## 🧪 Verification
1.  Open `http://localhost:3000` in your browser.
2.  Upload an ultrasound image.
3.  Check the backend terminal logs—you should see "Prediction saved to database".
