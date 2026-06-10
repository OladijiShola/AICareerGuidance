"""
AI Career Guidance & Roadmap System
Nigerian secondary school career counselling tool powered by a Random Forest classifier.

Flow: Pretest (8 quick questions) → Skills Assessment (sliders pre-filled) → Results
"""

import os
import csv
import threading
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import openpyxl
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

import pyttsx3
import speech_recognition as sr

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (Red · Navy · White)
# ─────────────────────────────────────────────
RED        = "#C1121F"   # primary action / accent
RED_HOVER  = "#8B0000"   # darker hover state
NAVY       = "#0D1B2A"   # deepest background
NAVY_MID   = "#1B263B"   # header / card surfaces
NAVY_LIGHT = "#2E4057"   # subtle card interior
WHITE      = "#FFFFFF"
OFF_WHITE  = "#F0F2F5"   # page background (light mode)
GREY_TEXT  = "#7B8FA1"   # muted labels

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CSV_FILE    = "students.csv"
MODEL_CACHE = "career_model.pkl"

UNIVERSITIES = {
    "University of Lagos (UNILAG)":      ["Computer Science", "Medicine and Surgery", "Law", "Accounting"],
    "Covenant University":                ["Computer Science", "Accounting", "Civil Engineering", "Visual Arts"],
    "Obafemi Awolowo University (OAU)":  ["Medicine and Surgery", "Law", "Civil Engineering", "Education"],
    "University of Ibadan (UI)":         ["Medicine and Surgery", "Law", "Education"],
    "Ahmadu Bello University (ABU)":     ["Civil Engineering", "Computer Science", "Medicine and Surgery"],
}

CAREERS = {
    "Software Engineer": {
        "course": "Computer Science / Computer Engineering",
        "jamb_subjects": "English, Mathematics, Physics, and Chemistry",
        "subjects": ["Mathematics", "Physics", "English", "Computer Studies"],
        "feature_weights": {
            "Mathematics": 5, "Physics": 4, "Computer Studies": 5,
            "Problem Solving": 5, "Creativity": 4,
        },
        "progression": [
            "Junior Developer / Intern (0–2 yrs)",
            "Senior Engineer / Tech Lead (3–6 yrs)",
            "Software Architect / CTO (7+ yrs)",
        ],
        "edu_milestones": [
            "B.Sc. Computer Science or B.Eng. Computer Engineering",
            "Specialisations: Cloud Architecture, AI Systems, Full-Stack",
        ],
        "alt_routes": [
            "ALX Nigeria Accelerator",
            "Decagon / Semicolon Tech Bootcamps",
            "AWS / Google Professional Cloud Certifications",
        ],
    },
    "Medical Doctor": {
        "course": "Medicine and Surgery",
        "jamb_subjects": "English, Physics, Chemistry, and Biology",
        "subjects": ["Biology", "Chemistry", "Physics", "English"],
        "feature_weights": {
            "Biology": 5, "Chemistry": 5, "Physics": 3,
            "Helping People": 5, "Problem Solving": 4,
        },
        "progression": [
            "House Officer / Intern (1 yr)",
            "Medical Officer / Resident (2–5 yrs)",
            "Consultant Specialist / Chief Medical Director (7+ yrs)",
        ],
        "edu_milestones": [
            "MBBS Degree (accredited Nigerian university)",
            "MDCAN Residency Training & Board Fellowships",
        ],
        "alt_routes": [
            "Master of Public Health (MPH)",
            "Health Technology & Hospital Administration diplomas",
        ],
    },
    "Lawyer": {
        "course": "Law (LL.B)",
        "jamb_subjects": "English, Literature in English, Government, and Religious Studies",
        "subjects": ["English", "Literature", "Government"],
        "feature_weights": {
            "Literature": 5, "Government": 5, "English": 5,
            "Communication": 5, "Leadership": 4,
        },
        "progression": [
            "Junior Associate (1–3 yrs)",
            "Senior Associate / Partner (5–9 yrs)",
            "Senior Advocate of Nigeria (SAN) / Judicial Bench",
        ],
        "edu_milestones": [
            "LL.B Degree (university)",
            "B.L. Professional Degree (Nigerian Law School)",
        ],
        "alt_routes": [
            "Institute of Chartered Secretaries & Administrators of Nigeria (ICSAN)",
            "CIArb Arbitration Tracks",
        ],
    },
    "Accountant": {
        "course": "Accounting",
        "jamb_subjects": "English, Mathematics, Economics, and a Financial/Social Science subject",
        "subjects": ["Mathematics", "Economics", "English"],
        "feature_weights": {
            "Mathematics": 5, "Economics": 5,
            "Problem Solving": 4, "Leadership": 3,
        },
        "progression": [
            "Audit / Accounts Trainee (0–2 yrs)",
            "Financial Analyst / Management Accountant (3–5 yrs)",
            "Chief Financial Officer (CFO) / Auditor General",
        ],
        "edu_milestones": [
            "B.Sc. Accounting or Banking & Finance",
        ],
        "alt_routes": [
            "ICAN Professional Licence (Nigeria)",
            "ACCA Certification",
            "FinTech & Forensic Data Analysis programmes",
        ],
    },
    "Graphic Designer": {
        "course": "Visual Arts / Creative Arts",
        "jamb_subjects": "English, Fine Arts, and two other Arts or Social Science subjects",
        "subjects": ["Fine Arts", "English", "Computer Studies"],
        "feature_weights": {
            "Fine Arts": 5, "Computer Studies": 3,
            "Creativity": 5, "Communication": 4,
        },
        "progression": [
            "Junior Designer / Freelancer (0–2 yrs)",
            "Art Director / UI/UX Lead (3–6 yrs)",
            "Creative Director / Agency Founder (7+ yrs)",
        ],
        "edu_milestones": [
            "B.A. Creative Arts / Fine Arts",
        ],
        "alt_routes": [
            "UI/UX Product Design Bootcamps",
            "Adobe Certified Professional tracks",
        ],
    },
    "Civil Engineer": {
        "course": "Civil Engineering",
        "jamb_subjects": "English, Mathematics, Physics, and Chemistry",
        "subjects": ["Mathematics", "Physics", "Chemistry"],
        "feature_weights": {
            "Mathematics": 5, "Physics": 5, "Chemistry": 4,
            "Problem Solving": 5, "Leadership": 3,
        },
        "progression": [
            "Graduate Engineer (0–2 yrs)",
            "Project Engineer / Site Manager (3–6 yrs)",
            "Principal Consultant / Structural Director",
        ],
        "edu_milestones": [
            "B.Eng. Civil Engineering",
            "NSE / COREN Professional Registration (Nigeria)",
        ],
        "alt_routes": [
            "Project Management Professional (PMP) Certification",
            "BIM Specialised Software Mastery",
        ],
    },
    "Teacher": {
        "course": "Education / Arts / Sciences",
        "jamb_subjects": "English, the core teaching subject, and two relevant school courses",
        "subjects": ["English", "Mathematics"],
        "feature_weights": {
            "English": 4, "Communication": 5,
            "Helping People": 5, "Leadership": 4,
        },
        "progression": [
            "Classroom Instructor (0–2 yrs)",
            "Head of Department / Vice Principal (3–6 yrs)",
            "School Administrator / Ministry of Education Consultant",
        ],
        "edu_milestones": [
            "B.Ed. Degree / B.Sc. with PGDE",
            "TRCN Certification (Teachers Registration Council of Nigeria)",
        ],
        "alt_routes": [
            "Educational Technology & Digital Learning Certifications",
            "Curriculum Development Framework tracks",
        ],
    },
}

FEATURE_NAMES = [
    "Mathematics", "Biology", "Chemistry", "Physics", "Computer Studies",
    "Economics", "Literature", "Government", "Fine Arts",
    "Problem Solving", "Leadership", "Communication", "Creativity", "Helping People",
]
CAREER_LIST = list(CAREERS.keys())

# ─────────────────────────────────────────────
#  PRETEST DEFINITION
#  Each question has options; each option maps
#  feature names → score adjustments (+1 to +3).
#  Adjustments are additive and clamped to [1,5].
# ─────────────────────────────────────────────
PRETEST_QUESTIONS = [
    {
        "question": "Which subject do you look forward to most in school?",
        "options": {
            "Mathematics & Numbers":     {"Mathematics": 2, "Economics": 1},
            "Sciences (Biology/Chem)":   {"Biology": 2, "Chemistry": 2},
            "English & Literature":      {"English": 2, "Literature": 2, "Communication": 1},
            "Arts & Creative subjects":  {"Fine Arts": 2, "Creativity": 2},
        },
    },
    {
        "question": "What kind of project would you most enjoy doing?",
        "options": {
            "Building an app or website":          {"Computer Studies": 2, "Problem Solving": 2, "Creativity": 1},
            "Designing a poster or logo":          {"Fine Arts": 2, "Creativity": 2, "Communication": 1},
            "Writing and presenting an argument":  {"Literature": 2, "Government": 1, "Communication": 2},
            "Running an experiment in the lab":    {"Biology": 2, "Chemistry": 2, "Physics": 1},
        },
    },
    {
        "question": "When you have free time, what do you naturally do?",
        "options": {
            "Tinker with gadgets or code":   {"Computer Studies": 2, "Problem Solving": 1, "Physics": 1},
            "Draw, design, or make things":  {"Fine Arts": 2, "Creativity": 2},
            "Read, debate, or write":        {"Literature": 2, "Communication": 2},
            "Help or teach people around you": {"Helping People": 3, "Communication": 1},
        },
    },
    {
        "question": "How do you prefer to solve a hard problem?",
        "options": {
            "Break it into logical steps and analyse": {"Problem Solving": 2, "Mathematics": 1},
            "Research and gather as much info as possible": {"Biology": 1, "Chemistry": 1, "Problem Solving": 1},
            "Discuss it with others and get consensus": {"Communication": 2, "Leadership": 1},
            "Sketch ideas and think visually":           {"Creativity": 2, "Fine Arts": 1},
        },
    },
    {
        "question": "Which of these roles appeals to you most?",
        "options": {
            "Doctor or health professional":   {"Biology": 2, "Chemistry": 1, "Helping People": 2},
            "Lawyer or public advocate":       {"Government": 2, "Communication": 2, "Leadership": 1},
            "Engineer or builder":             {"Mathematics": 2, "Physics": 2, "Problem Solving": 1},
            "Teacher or mentor":               {"Helping People": 2, "Communication": 2, "Leadership": 1},
        },
    },
    {
        "question": "What kind of work environment excites you?",
        "options": {
            "Office with numbers and reports":   {"Mathematics": 1, "Economics": 2, "Problem Solving": 1},
            "Studio or creative workspace":      {"Fine Arts": 2, "Creativity": 2},
            "Hospital, clinic, or field work":   {"Biology": 2, "Helping People": 2},
            "Courtroom, boardroom, or outdoors": {"Government": 1, "Leadership": 2, "Communication": 1},
        },
    },
    {
        "question": "Which best describes your strongest skill?",
        "options": {
            "I'm good with numbers and logic": {"Mathematics": 2, "Problem Solving": 2},
            "I communicate and persuade well": {"Communication": 2, "Literature": 1, "Leadership": 1},
            "I'm creative and visually minded": {"Creativity": 2, "Fine Arts": 2},
            "I'm empathetic and people-focused": {"Helping People": 2, "Communication": 1},
        },
    },
    {
        "question": "What motivates you most about a future career?",
        "options": {
            "Solving technical challenges":   {"Problem Solving": 2, "Mathematics": 1, "Physics": 1},
            "Creating something beautiful":   {"Creativity": 2, "Fine Arts": 2},
            "Making a difference in people's lives": {"Helping People": 2, "Leadership": 1},
            "Building wealth and business":   {"Economics": 2, "Mathematics": 1, "Leadership": 1},
        },
    },
]


# ─────────────────────────────────────────────
#  ML ENGINE
# ─────────────────────────────────────────────
class MLEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_CACHE):
            try:
                self.model = joblib.load(MODEL_CACHE)
                return
            except Exception:
                pass

        X, y = [], []
        rng = np.random.default_rng(42)
        for _ in range(300):
            for idx, (_, meta) in enumerate(CAREERS.items()):
                weights = meta["feature_weights"]
                row = [int(np.clip(weights.get(f, 2) + rng.integers(-1, 2), 1, 5))
                       for f in FEATURE_NAMES]
                X.append(row)
                y.append(idx)

        self.model.fit(X, y)
        try:
            joblib.dump(self.model, MODEL_CACHE)
        except Exception:
            pass

    def predict(self, feature_values: list) -> tuple:
        probs = self.model.predict_proba([feature_values])[0]
        best_idx = int(np.argmax(probs))
        career = CAREER_LIST[best_idx]
        confidence = round(float(probs[best_idx]) * 100, 1)
        prob_map = {CAREER_LIST[i]: float(p) for i, p in enumerate(probs)}
        return career, confidence, prob_map


# ─────────────────────────────────────────────
#  TTS ENGINE
# ─────────────────────────────────────────────
class TTSEngine:
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def _get_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 145)
        return self._engine

    def speak(self, text: str):
        def _worker():
            with self._lock:
                try:
                    self._get_engine().say(text)
                    self._get_engine().runAndWait()
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────
#  DATA PERSISTENCE
# ─────────────────────────────────────────────
class StudentLog:
    HEADER = (
        ["Name", "Class", "Age", "Predicted Career", "Confidence (%)",
         "Recommended Course", "Matching Universities", "Date"]
        + FEATURE_NAMES
    )

    def append(self, name, school, age, career, confidence, course, unis, vector):
        write_header = not os.path.exists(CSV_FILE)
        try:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(self.HEADER)
                w.writerow([name, school, age, career, confidence, course, unis,
                            datetime.now().strftime("%Y-%m-%d")] + vector)
        except IOError:
            messagebox.showerror(
                "Save Error",
                "Could not write to the student log.\n"
                "Make sure students.csv is not open in another programme.",
            )

    @staticmethod
    def load_dataframe():
        if not os.path.exists(CSV_FILE):
            return None
        return pd.read_csv(CSV_FILE)


# ─────────────────────────────────────────────
#  PRETEST WINDOW
# ─────────────────────────────────────────────
class PretestWindow(ctk.CTkToplevel):
    """
    An 8-question multiple-choice pretest shown before the skills sliders.
    On completion it calls on_complete(score_boosts) where score_boosts is
    a dict of {feature: total_adjustment}.
    """

    def __init__(self, parent, on_complete):
        super().__init__(parent)
        self.title("Quick Interest Check")
        self.geometry("620x460")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.on_complete = on_complete
        self.questions   = PRETEST_QUESTIONS
        self.total       = len(self.questions)
        self.current_idx = 0
        self.boosts: dict = {f: 0 for f in FEATURE_NAMES}
        self.selected_var = tk.StringVar()

        self._build()
        self._load_question()

    def _build(self):
        # Header strip
        hdr = ctk.CTkFrame(self, corner_radius=0, height=56,
                           fg_color=(NAVY_MID, NAVY))
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="Interest & Aptitude Check",
            text_color=WHITE,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        ).pack(side="left", padx=20, pady=10)

        self.step_lbl = ctk.CTkLabel(
            hdr, text="", text_color=WHITE,
            font=ctk.CTkFont(size=12),
        )
        self.step_lbl.pack(side="right", padx=20)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self, height=6,
                                           fg_color=(NAVY_LIGHT, NAVY_LIGHT),
                                           progress_color=RED)
        self.progress.pack(fill="x", padx=0, pady=0)
        self.progress.set(0)

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=30, pady=20)

        self.q_label = ctk.CTkLabel(
            body, text="",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            wraplength=540, justify="left",
        )
        self.q_label.pack(anchor="w", pady=(0, 18))

        self.option_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.option_frame.pack(fill="both", expand=True)

        # Footer buttons
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=30, pady=15)

        self.skip_btn = ctk.CTkButton(
            footer, text="Skip pretest →",
            fg_color="transparent",
            border_color=(NAVY_LIGHT, NAVY_LIGHT),
            border_width=1,
            text_color=(NAVY_MID, WHITE),
            hover_color=(OFF_WHITE, NAVY_LIGHT),
            width=130,
            command=self._skip,
        )
        self.skip_btn.pack(side="left")

        self.next_btn = ctk.CTkButton(
            footer, text="Next  →",
            fg_color=RED, hover_color=RED_HOVER,
            text_color=WHITE,
            font=ctk.CTkFont(weight="bold"),
            width=120,
            command=self._advance,
            state="disabled",
        )
        self.next_btn.pack(side="right")

    def _load_question(self):
        q = self.questions[self.current_idx]
        self.selected_var.set("")
        self.next_btn.configure(state="disabled")

        self.step_lbl.configure(
            text=f"Question {self.current_idx + 1} of {self.total}"
        )
        self.progress.set((self.current_idx) / self.total)
        self.q_label.configure(text=q["question"])

        for child in self.option_frame.winfo_children():
            child.destroy()

        for label in q["options"]:
            rb = ctk.CTkRadioButton(
                self.option_frame,
                text=label,
                variable=self.selected_var,
                value=label,
                font=ctk.CTkFont(size=13),
                fg_color=RED,
                hover_color=RED_HOVER,
                command=lambda: self.next_btn.configure(state="normal"),
            )
            rb.pack(anchor="w", padx=10, pady=6)

    def _advance(self):
        chosen = self.selected_var.get()
        if not chosen:
            return
        adjustments = self.questions[self.current_idx]["options"][chosen]
        for feat, delta in adjustments.items():
            self.boosts[feat] = self.boosts.get(feat, 0) + delta

        self.current_idx += 1
        if self.current_idx < self.total:
            self._load_question()
        else:
            self._finish()

    def _finish(self):
        self.progress.set(1.0)
        self.grab_release()
        self.destroy()
        self.on_complete(self.boosts)

    def _skip(self):
        self.grab_release()
        self.destroy()
        self.on_complete({})   # no boosts — sliders stay at default 3


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class CareerGuidanceApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("AI Career Guidance & Roadmap System")
        self.geometry("1350x880")
        self.minsize(1100, 700)

        self.passport_path = None
        self.ml_engine  = MLEngine()
        self.tts        = TTSEngine()
        self.student_log = StudentLog()
        self.feature_vars = {feat: tk.IntVar(value=3) for feat in FEATURE_NAMES}

        self._build_ui()

        # Launch pretest automatically on startup (after window is ready)
        self.after(300, self._open_pretest)

    # ── Layout ──────────────────────────────

    def _build_ui(self):
        self._build_header()

        self.input_panel = ctk.CTkScrollableFrame(
            self, label_text="  Student Profile & Skills Assessment",
            label_font=ctk.CTkFont(size=13, weight="bold"),
            label_fg_color=(NAVY_MID, NAVY),
            label_text_color=WHITE,
        )
        self.input_panel.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.output_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.output_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self._build_input_form()
        self._build_output_tabs()

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0, height=70,
                              fg_color=(NAVY_MID, NAVY))
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Red accent bar at very top
        accent_bar = ctk.CTkFrame(self, corner_radius=0, height=4,
                                  fg_color=(RED, RED))
        accent_bar.pack(fill="x", side="top", before=header)

        ctk.CTkLabel(
            header,
            text="AI CAREER GUIDANCE & ROADMAP SYSTEM",
            text_color=WHITE,
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        ).pack(side="left", padx=30, pady=15)

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.pack(side="right", padx=30)

        self.dark_switch = ctk.CTkSwitch(
            controls, text="Dark mode", text_color=WHITE,
            button_color=RED, button_hover_color=RED_HOVER,
            progress_color=NAVY_LIGHT,
            command=self._toggle_theme,
        )
        self.dark_switch.pack(side="left", padx=15)

        self.voice_btn = ctk.CTkButton(
            controls, text="🎙️ Listen", width=100,
            fg_color=NAVY_LIGHT, hover_color=RED,
            text_color=WHITE,
            font=ctk.CTkFont(weight="bold"),
            command=self._start_voice_listener,
        )
        self.voice_btn.pack(side="left", padx=8)

        ctk.CTkButton(
            controls, text="📊 Dashboard", width=120,
            fg_color=NAVY_LIGHT, hover_color=RED,
            text_color=WHITE,
            font=ctk.CTkFont(weight="bold"),
            command=self._open_dashboard,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            controls, text="🔄 Retake Pretest", width=140,
            fg_color=RED, hover_color=RED_HOVER,
            text_color=WHITE,
            font=ctk.CTkFont(weight="bold"),
            command=self._open_pretest,
        ).pack(side="left", padx=8)

    def _build_input_form(self):
        # ── Profile card ──
        profile = ctk.CTkFrame(self.input_panel,
                               border_width=1, border_color=(NAVY_LIGHT, NAVY_LIGHT))
        profile.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            profile, text="Student Information",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=(NAVY_MID, WHITE),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=10)

        fields = [
            ("Full Name",      "name_entry",  "E.g. Tolani Benson", 220),
            ("Current Class",  "class_entry", "E.g. SS3",           220),
            ("Age",            "age_entry",   "E.g. 17",            100),
        ]
        for row_idx, (label, attr, placeholder, width) in enumerate(fields, start=1):
            ctk.CTkLabel(profile, text=label).grid(
                row=row_idx, column=0, sticky="w", padx=15, pady=5
            )
            entry = ctk.CTkEntry(profile, width=width, placeholder_text=placeholder,
                                 border_color=(RED, RED), border_width=1)
            entry.grid(row=row_idx, column=1, sticky="w", padx=15, pady=5)
            setattr(self, attr, entry)

        self.photo_label = ctk.CTkLabel(
            profile, text="No photo\nuploaded",
            width=120, height=120,
            fg_color=(OFF_WHITE, NAVY_LIGHT),
            corner_radius=8,
        )
        self.photo_label.grid(row=1, column=2, rowspan=3, padx=25, pady=10)

        ctk.CTkButton(
            profile, text="Upload Photo", width=110, height=28,
            fg_color=NAVY_MID, hover_color=RED, text_color=WHITE,
            font=ctk.CTkFont(size=12),
            command=self._upload_photo,
        ).grid(row=4, column=2, pady=5, padx=25)

        # ── Pretest status banner ──
        self.pretest_banner = ctk.CTkFrame(
            self.input_panel, fg_color=(OFF_WHITE, NAVY_LIGHT),
            border_width=1, border_color=(RED, RED), corner_radius=8,
        )
        self.pretest_banner.pack(fill="x", padx=10, pady=(0, 4))

        self.pretest_status_lbl = ctk.CTkLabel(
            self.pretest_banner,
            text="⚪  Pretest not yet completed — sliders are at default (3).",
            font=ctk.CTkFont(size=12),
            text_color=GREY_TEXT,
        )
        self.pretest_status_lbl.pack(side="left", padx=12, pady=8)

        # ── Skills sliders card ──
        skills = ctk.CTkFrame(self.input_panel,
                              border_width=1, border_color=(NAVY_LIGHT, NAVY_LIGHT))
        skills.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            skills,
            text="Fine-tune your subject & strength scores  (1 = Low · 5 = High)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=(NAVY_MID, WHITE),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=15, pady=10)

        for idx, (feat, var) in enumerate(self.feature_vars.items()):
            row = (idx // 2) + 1
            col = (idx % 2) * 2
            ctk.CTkLabel(skills, text=feat, anchor="w").grid(
                row=row, column=col, sticky="w", padx=15, pady=7
            )
            ctk.CTkSegmentedButton(
                skills, values=["1", "2", "3", "4", "5"], variable=var,
                selected_color=RED, selected_hover_color=RED_HOVER,
                unselected_color=(NAVY_LIGHT, NAVY_LIGHT),
                unselected_hover_color=(NAVY_MID, NAVY_MID),
                text_color=WHITE,
            ).grid(row=row, column=col + 1, padx=10, pady=7, sticky="e")

        ctk.CTkButton(
            self.input_panel,
            text="🚀  Get Career Recommendation",
            height=48,
            fg_color=RED, hover_color=RED_HOVER,
            text_color=WHITE,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=8,
            command=self._run_prediction,
        ).pack(fill="x", padx=10, pady=20)

    def _build_output_tabs(self):
        self.tabs = ctk.CTkTabview(
            self.output_panel, corner_radius=10,
            segmented_button_fg_color=(NAVY_MID, NAVY),
            segmented_button_selected_color=RED,
            segmented_button_selected_hover_color=RED_HOVER,
            segmented_button_unselected_color=(NAVY_MID, NAVY),
            segmented_button_unselected_hover_color=NAVY_LIGHT,
            text_color=WHITE,
        )
        self.tabs.pack(fill="both", expand=True)

        tab_results   = self.tabs.add("📊 Results")
        tab_roadmap   = self.tabs.add("🗺️ Career Roadmap")
        tab_education = self.tabs.add("🎓 Education Path")

        self.results_box = ctk.CTkTextbox(
            tab_results, height=250,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8,
            border_color=(NAVY_LIGHT, NAVY_LIGHT), border_width=1,
        )
        self.results_box.pack(fill="x", padx=10, pady=10)

        self.chart_frame = ctk.CTkFrame(tab_results,
                                        border_width=1, border_color=(NAVY_LIGHT, NAVY_LIGHT))
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            self.chart_frame,
            text="Run an assessment to see the career probability chart.",
            text_color=GREY_TEXT,
        ).pack(expand=True)

        self.roadmap_frame = ctk.CTkScrollableFrame(
            tab_roadmap, label_text="Career Progression Milestones",
            label_font=ctk.CTkFont(size=13, weight="bold"),
            label_fg_color=(NAVY_MID, NAVY), label_text_color=WHITE,
        )
        self.roadmap_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(self.roadmap_frame,
                     text="Complete an assessment to see the career roadmap.",
                     text_color=GREY_TEXT).pack(pady=40)

        self.education_frame = ctk.CTkScrollableFrame(
            tab_education, label_text="Academic Requirements & Alternative Pathways",
            label_font=ctk.CTkFont(size=13, weight="bold"),
            label_fg_color=(NAVY_MID, NAVY), label_text_color=WHITE,
        )
        self.education_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(self.education_frame,
                     text="Complete an assessment to see the education pathway.",
                     text_color=GREY_TEXT).pack(pady=40)

    # ── Pretest ─────────────────────────────

    def _open_pretest(self):
        PretestWindow(self, on_complete=self._apply_pretest_boosts)

    def _apply_pretest_boosts(self, boosts: dict):
        """
        Receive the cumulative feature boosts from the pretest and
        update all slider IntVars accordingly, clamped to [1, 5].
        """
        if not boosts:
            self.pretest_status_lbl.configure(
                text="⚪  Pretest skipped — you can adjust scores manually below.",
                text_color=GREY_TEXT,
            )
            return

        for feat, var in self.feature_vars.items():
            base  = 2          # start from 2 (slightly below neutral)
            delta = boosts.get(feat, 0)
            new_val = int(np.clip(base + delta, 1, 5))
            var.set(new_val)

        self.pretest_status_lbl.configure(
            text="✅  Pretest complete — scores pre-filled below. Adjust freely.",
            text_color=("#1a7a3c", "#4caf7a"),
        )

    # ── Actions ─────────────────────────────

    def _toggle_theme(self):
        ctk.set_appearance_mode("Dark" if self.dark_switch.get() else "Light")

    def _upload_photo(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.passport_path = path
            img   = Image.open(path)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(120, 120))
            self.photo_label.configure(image=photo, text="")

    def _start_voice_listener(self):
        def _worker():
            recogniser = sr.Recognizer()
            with sr.Microphone() as source:
                self.voice_btn.configure(text="Listening…", fg_color=RED)
                try:
                    audio = recogniser.listen(source, timeout=4, phrase_time_limit=5)
                    cmd   = recogniser.recognize_google(audio).lower()
                    if any(w in cmd for w in ("analyse", "analyze", "predict", "run")):
                        self.after(0, self._run_prediction)
                except Exception:
                    pass
                finally:
                    self.voice_btn.configure(text="🎙️ Listen", fg_color=NAVY_LIGHT)
        threading.Thread(target=_worker, daemon=True).start()

    def _validate_inputs(self) -> bool:
        name  = self.name_entry.get().strip()
        school = self.class_entry.get().strip()
        age_raw = self.age_entry.get().strip()

        if not name or not school:
            messagebox.showwarning(
                "Missing Information",
                "Please enter the student's name and current class before continuing.",
            )
            return False

        if age_raw:
            try:
                if not (10 <= int(age_raw) <= 65):
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid Age",
                                       "Please enter a valid age between 10 and 65.")
                return False
        return True

    def _run_prediction(self):
        if not self._validate_inputs():
            return

        name   = self.name_entry.get().strip()
        school = self.class_entry.get().strip()
        age    = self.age_entry.get().strip() or "N/A"

        vector = [int(self.feature_vars[f].get()) for f in FEATURE_NAMES]
        career, confidence, prob_map = self.ml_engine.predict(vector)
        meta = CAREERS[career]

        subjects_str   = ", ".join(meta["subjects"])
        matched_unis   = [u for u, courses in UNIVERSITIES.items()
                          if any(c in meta["course"] for c in courses)]
        unis_block     = "\n  • ".join(matched_unis) if matched_unis \
                         else "See alternative technical institutions"

        report = (
            f"{'═' * 66}\n"
            f"  CAREER ASSESSMENT REPORT\n"
            f"{'═' * 66}\n"
            f"  Student:     {name}\n"
            f"  Class:       {school}\n"
            f"  Age:         {age}\n"
            f"  Date:        {datetime.now().strftime('%d %b %Y, %H:%M')}\n"
            f"{'─' * 66}\n"
            f"  Predicted Career:     {career}  ({confidence}% confidence)\n"
            f"  Recommended Course:   {meta['course']}\n"
            f"  Key Subjects:         {subjects_str}\n"
            f"  JAMB Combination:     {meta['jamb_subjects']}\n"
            f"{'─' * 66}\n"
            f"  Matching Universities:\n  • {unis_block}\n"
            f"{'═' * 66}\n"
        )

        self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, report)

        self._render_chart(prob_map)
        self._render_roadmap(career)
        self._render_education(career)

        self.student_log.append(name, school, age, career, confidence,
                                meta["course"], ", ".join(matched_unis), vector)
        self._save_pdf(name, report)
        self.tts.speak(
            f"Assessment complete. The recommended career is {career}."
        )

    # ── Output rendering ────────────────────

    def _render_chart(self, prob_map: dict):
        for child in self.chart_frame.winfo_children():
            child.destroy()

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg = "#1B263B" if is_dark else "#F0F2F5"
        fg = WHITE if is_dark else NAVY

        fig, ax = plt.subplots(figsize=(6, 2.8))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        labels   = list(prob_map.keys())
        values   = [v * 100 for v in prob_map.values()]
        max_val  = max(values)
        colors   = [RED if v == max_val else "#8B9BB4" for v in values]

        ax.bar(labels, values, color=colors, width=0.5)
        ax.set_title("Career Probability Breakdown", color=fg,
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability (%)", color=fg, fontsize=9)
        ax.tick_params(colors=fg, labelsize=8)
        ax.set_ylim(0, 100)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(fg)
        ax.spines["bottom"].set_color(fg)

        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _render_roadmap(self, career: str):
        for child in self.roadmap_frame.winfo_children():
            child.destroy()

        meta = CAREERS[career]
        ctk.CTkLabel(
            self.roadmap_frame,
            text=f"🎯  {career.upper()} — Career Progression",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=RED,
        ).pack(anchor="w", padx=10, pady=10)

        for idx, step in enumerate(meta["progression"]):
            node = ctk.CTkFrame(self.roadmap_frame,
                                fg_color=(OFF_WHITE, NAVY_LIGHT))
            node.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(
                node, text=f"  Level {idx + 1}  ",
                fg_color=NAVY_MID, text_color=WHITE, corner_radius=4,
            ).pack(side="left", padx=15, pady=10)

            ctk.CTkLabel(node, text=step, font=ctk.CTkFont(size=13)
                         ).pack(side="left", padx=10)

            if idx < len(meta["progression"]) - 1:
                ctk.CTkLabel(
                    self.roadmap_frame, text="  │\n  ▼",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=GREY_TEXT,
                ).pack(anchor="w", padx=45)

    def _render_education(self, career: str):
        for child in self.education_frame.winfo_children():
            child.destroy()

        meta = CAREERS[career]

        ctk.CTkLabel(
            self.education_frame,
            text="🎓  Required Degrees & Certifications",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=NAVY_MID,
        ).pack(anchor="w", padx=10, pady=10)

        for degree in meta["edu_milestones"]:
            f = ctk.CTkFrame(self.education_frame,
                             border_width=1, border_color=(NAVY_LIGHT, NAVY_LIGHT))
            f.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(f, text=f"• {degree}", font=ctk.CTkFont(size=12),
                         wraplength=480, justify="left").pack(anchor="w", padx=10, pady=8)

        ctk.CTkLabel(
            self.education_frame,
            text="🚀  Local Accelerators & Alternative Pathways",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=RED,
        ).pack(anchor="w", padx=10, pady=12)

        for route in meta["alt_routes"]:
            f = ctk.CTkFrame(self.education_frame,
                             border_width=1, border_color=(RED, RED))
            f.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(f, text=f"⚡ {route}", font=ctk.CTkFont(size=12),
                         wraplength=480, justify="left").pack(anchor="w", padx=10, pady=8)

    def _save_pdf(self, name: str, body_text: str):
        filename = filedialog.asksaveasfilename(
            initialfile=f"{name.replace(' ', '_')}_CareerReport.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf")],
        )
        if not filename:
            return

        doc    = SimpleDocTemplate(filename)
        styles = getSampleStyleSheet()
        story  = []

        if os.path.exists("logo.png"):
            story.append(RLImage("logo.png", width=140, height=45))
            story.append(Spacer(1, 10))

        story.append(
            Paragraph(f"<b>Career Assessment Report — {name.upper()}</b>",
                      styles["Title"])
        )
        story.append(Spacer(1, 15))

        if self.passport_path and os.path.exists(self.passport_path):
            story.append(RLImage(self.passport_path, width=100, height=100))
            story.append(Spacer(1, 15))

        normal = styles["Normal"]
        normal.fontName = "Helvetica"
        normal.leading  = 16

        for line in body_text.split("\n"):
            safe = (line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            story.append(Paragraph(safe if safe.strip() else "&nbsp;", normal))

        try:
            doc.build(story)
        except Exception as exc:
            messagebox.showerror("PDF Error", f"Could not save the PDF report:\n{exc}")

    # ── Dashboard ───────────────────────────

    def _open_dashboard(self):
        df = StudentLog.load_dataframe()
        if df is None:
            messagebox.showwarning(
                "No Data Yet",
                "No student records found. Run at least one assessment first.",
            )
            return

        dash = ctk.CTkToplevel(self)
        dash.title("Student Cohort Dashboard")
        dash.geometry("1050x650")
        dash.transient(self)
        dash.grab_set()

        # Header
        hdr = ctk.CTkFrame(dash, height=60, corner_radius=0,
                           fg_color=(NAVY_MID, NAVY))
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text=f"📊  Student Records: {len(df)}",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=WHITE,
        ).pack(side="left", padx=20)

        ctk.CTkButton(
            hdr, text="📥 Export to Excel",
            fg_color=RED, hover_color=RED_HOVER, text_color=WHITE,
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self._export_excel(df),
        ).pack(side="right", padx=20, pady=10)

        chart_area = ctk.CTkFrame(dash)
        chart_area.pack(fill="both", expand=True, padx=15, pady=10)

        is_dark  = ctk.get_appearance_mode() == "Dark"
        panel_bg = "#1B263B" if is_dark else "#F0F2F5"
        txt_fg   = WHITE if is_dark else NAVY

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor(panel_bg)
        ax1.set_facecolor(panel_bg)
        ax2.set_facecolor(panel_bg)

        df["Predicted Career"].value_counts().plot(kind="bar", ax=ax1, color=RED)
        ax1.set_title("Career Distribution", color=txt_fg, fontsize=11, fontweight="bold")
        ax1.set_xlabel("")
        ax1.tick_params(colors=txt_fg, labelsize=8)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

        df[FEATURE_NAMES].mean().plot(
            kind="line", marker="s", color=NAVY_MID if not is_dark else "#8B9BB4",
            linewidth=2.5, ax=ax2,
        )
        ax2.set_title("Average Scores Across Cohort", color=txt_fg,
                      fontsize=11, fontweight="bold")
        ax2.tick_params(colors=txt_fg, labelsize=8)
        ax2.set_xticks(range(len(FEATURE_NAMES)))
        ax2.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        plt.close(fig)

    @staticmethod
    def _export_excel(df: pd.DataFrame):
        filename = f"CareerGuidanceExport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Student Evaluations", index=False)
            messagebox.showinfo("Export Complete", f"Spreadsheet saved:\n{filename}")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not write Excel file:\n{exc}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = CareerGuidanceApp()
    app.mainloop()