from __future__ import annotations
import os
import sys
import io
import math
import tempfile
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import requests
import json
from datetime import datetime, timezone

# Optional PDF support
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

# Optional Statistics support
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

# OMR / CV stack
import cv2
import numpy as np
from sklearn.cluster import KMeans

# -------------------------------
# JEE Mains configuration (default rules)
# -------------------------------
SUBJECTS = {
    "Physics":  (1, 25),   # Q01..Q25
    "Chemistry":(26, 50),  # Q26..Q50
    "Maths":    (51, 75),  # Q51..Q75
}

MARK_CORRECT = 4.0
MARK_WRONG   = -1.0
MARK_SKIP    = 0.0

TOTAL_PER_SUBJECT = 25 * MARK_CORRECT  # 100
TOTAL_MAX = len(SUBJECTS) * TOTAL_PER_SUBJECT  # 300

# -------------------------------
# OMR Processor (image -> answers)
# -------------------------------
class OMRProcessor:
    def __init__(self):
        self.SHOW_DEBUG = False  # UI toggles this

    # ========= Helpers =========
    @staticmethod
    def safe_kmeans_1d(values, k, random_state=42, n_init=10):
        if len(values) < k:
            return None, None
        arr = np.array(values, dtype=np.float32).reshape(-1, 1)
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(arr)
        centers = km.cluster_centers_.flatten()
        return labels, centers

    @staticmethod
    def fill_ratio_for_contour(thresh, cnt):
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask_pixels = cv2.countNonZero(mask)
        if mask_pixels == 0:
            return 0.0
        filled = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        return float(filled) / float(mask_pixels)

    @staticmethod
    def format_numeric_raw_to_display(raw: str) -> str:
        if raw == "******":
            return "******"
        if isinstance(raw, str) and len(raw) == 6 and raw.isdigit():
            return raw[:4] + "." + raw[4:]
        return "null"

    @staticmethod
    def format_numeric_raw_to_real(raw: str) -> str:
        if raw == "******":
            return "******"
        if isinstance(raw, str) and len(raw) == 6 and raw.isdigit():
            val = int(raw)
            s = f"{val/100:.2f}"
            s = s.rstrip('0').rstrip('.')
            return s if s != "" else "0"
        return "null"

    @staticmethod
    def parse_numeric_raw_to_float(raw: str) -> Optional[float]:
        if raw == "******":
            return None
        if isinstance(raw, str) and len(raw) == 6 and raw.isdigit():
            return int(raw) / 100.0
        return None

    def _dbg_show(self, name, img):
        if self.SHOW_DEBUG:
            cv2.imshow(name, img)
            cv2.waitKey(0)

    # ========= Core steps =========
    def _extract_rollno_digits(self, roi,
                               roi_size=(680, 360),
                               area_min=80,
                               area_max=None,
                               aspect_ratio_range=(0.65, 1.5),
                               rows=6,
                               cols=10,
                               min_total_candidates=45,
                               min_fill_ratio=0.40,
                               min_margin=0.12,
                               debug_name=None,
                               is_rollno=False) -> str:
        if is_rollno:
            # BME roll number logic
            # You must pass the full image and ROI coords for roll number
            # This expects roi to be a tuple: (image, roi_coords)
            image, roi_coords = roi
            x1, y1, x2, y2 = roi_coords
            rollno = image[y1:y2, x1:x2]
            rollno = cv2.resize(rollno, roi_size)
            gray = cv2.cvtColor(rollno, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bubbled_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 60:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 1.4:
                        bubbled_contours.append(cnt)
            bubbles = []
            for cnt in bubbled_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                mask = np.zeros(thresh.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = cv2.mean(thresh, mask=mask)[0]
                bubbles.append({'cnt': cnt, 'cx': cx, 'cy': cy, 'mean': mean_val, 'bbox': (x, y, w, h)})
            if len(bubbles) < 30:
                return "******"
            cy_array = np.array([[b['cy']] for b in bubbles])
            kmeans_row = KMeans(n_clusters=6, n_init=10, random_state=42).fit(cy_array)
            row_labels = kmeans_row.labels_
            row_centers = kmeans_row.cluster_centers_.flatten()
            cx_array = np.array([[b['cx']] for b in bubbles])
            kmeans_col = KMeans(n_clusters=10, n_init=10, random_state=42).fit(cx_array)
            col_labels = kmeans_col.labels_
            col_centers = kmeans_col.cluster_centers_.flatten()
            for i, b in enumerate(bubbles):
                b['row'] = row_labels[i]
                b['col'] = col_labels[i]
            row_order = np.argsort(row_centers)
            col_order = np.argsort(col_centers)
            grid = [[None for _ in range(10)] for _ in range(6)]
            for b in bubbles:
                row_idx = np.where(row_order == b['row'])[0][0]
                col_idx = np.where(col_order == b['col'])[0][0]
                grid[row_idx][col_idx] = b
            roll_digits = []
            for row_idx in range(6):
                max_mean = -1
                digit = None
                for col_idx in range(10):
                    b = grid[row_idx][col_idx]
                    if b is None:
                        continue
                    if b['mean'] > max_mean:
                        max_mean = b['mean']
                        digit = col_idx
                        
                roll_digits.append(str(digit) if digit is not None else '?')

            roll_number = ''.join(roll_digits)
            if roll_number in ["888888", "999999", "666666"]:
                roll_number = "ENTER ROLL"

            return roll_number

        else:
            # Existing numeric logic
            rollno = cv2.resize(roi, roi_size)
            gray = cv2.cvtColor(rollno, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            roiA = roi_size[0] * roi_size[1]
            if area_max is None:
                area_max = int(0.05 * roiA)
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < area_min or area > area_max:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                ar = (w + 1e-6) / (h + 1e-6)
                if not (aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]):
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity < 0.4:
                    continue
                ratio = self.fill_ratio_for_contour(thresh, cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                candidates.append({"cnt": cnt, "cx": cx, "cy": cy, "ratio": ratio})
            if len(candidates) < max(min_total_candidates, rows + cols):
                return "******"
            row_labels, row_centers = self.safe_kmeans_1d([c["cy"] for c in candidates], rows)
            col_labels, col_centers = self.safe_kmeans_1d([c["cx"] for c in candidates], cols)
            if row_labels is None or col_labels is None:
                return "******"
            for i, c in enumerate(candidates):
                c["row"] = row_labels[i]
                c["col"] = col_labels[i]
            row_order = np.argsort(row_centers)
            col_order = np.argsort(col_centers)
            grid = [[None for _ in range(cols)] for _ in range(rows)]
            for c in candidates:
                r = int(np.where(row_order == c["row"])[0][0])
                k = int(np.where(col_order == c["col"])[0][0])
                prev = grid[r][k]
                if prev is None or c["ratio"] > prev["ratio"]:
                    grid[r][k] = c
            digits = []
            for r in range(rows):
                col_ratios = [grid[r][k]["ratio"] if grid[r][k] is not None else 0.0 for k in range(cols)]
                best_idx = int(np.argmax(col_ratios))
                best = col_ratios[best_idx]
                second = sorted(col_ratios, reverse=True)[1] if cols > 1 else 0.0
                if best < min_fill_ratio or (best - second) < min_margin :
                    return "******"
                digits.append(str(best_idx))
            if debug_name and self.SHOW_DEBUG:
                debug_vis = rollno.copy()
                for c in candidates:
                    cv2.drawContours(debug_vis, [c["cnt"]], -1, (0, 255, 0), 1)
                self._dbg_show(debug_name, debug_vis)
            return "".join(digits)

    def _detect_answers(self,
                        roi,
                        num_questions=20,
                        num_options=4,
                        bubble_area_min=300,
                        bubble_area_max=6000,
                        min_mark_ratio=0.4,
                        ambig_margin=0.06
                       ) -> List[str]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 0.7)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_contours, centers = [], []
        debug_bubble_vis = roi.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < bubble_area_min or area > bubble_area_max:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < 0.45:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bubble_contours.append(cnt)
            centers.append([cx, cy])
            if self.SHOW_DEBUG:
                cv2.drawContours(debug_bubble_vis, [cnt], -1, (0, 255, 0), 2)
        self._dbg_show("Detected Bubbles", debug_bubble_vis)

        expected = num_questions * num_options
        if len(centers) < max(expected // 2, num_questions):
            # Not enough bubbles detected; return placeholders
            return ["X"] * num_questions

        labels, row_centers = self.safe_kmeans_1d([c[1] for c in centers], num_questions, random_state=0, n_init=10)
        if labels is None:
            return ["X"] * num_questions

        temp_rows = {}
        for idx, label in enumerate(labels):
            temp_rows.setdefault(label, []).append((bubble_contours[idx], centers[idx]))

        sorted_labels = sorted(temp_rows.keys(),
                               key=lambda l: np.mean([center[1] for _, center in temp_rows[l]]))
        rows = [sorted(temp_rows[label], key=lambda b: b[1][0]) for label in sorted_labels]

        answers = []
        for q, bubbles in enumerate(rows):
            if len(bubbles) != num_options:
                answers.append("X")
                continue

            ratios = [self.fill_ratio_for_contour(thresh, cnt) for cnt, _ in bubbles]

            best_idx = int(np.argmax(ratios))
            best = ratios[best_idx]
            second = sorted(ratios, reverse=True)[1]

            if best < min_mark_ratio or (best - second) < ambig_margin:
                answers.append("null")
            else:
                opt = chr(ord("A") + best_idx)
                answers.append(opt)

        return answers

    # ========= Public API =========
    def process_image(self, path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a single OMR image, return (roll_number, answers dict Qxx -> value)
        MCQs are 'A'/'B'/'C'/'D' or 'null'/'X'. Numerics are float or None.
        """
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.resize(image, (1800, 2000))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise RuntimeError("No contours found. Check image quality/path.")

        biggest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)

        cropped = image[y:y + h, x:x + w]
        cropped_resized = cv2.resize(cropped, (1800, 2000))

        # ROIs
        rollno_roi = cropped_resized[0:224, 0:485]
        answer_roi = cropped_resized[214:1078, 0:1800]
        numeric_roi = cropped_resized[1070:1993, 0:1800]

        h_ans, w_ans = answer_roi.shape[:2]
        ans_roi1 = answer_roi[:, 0:w_ans // 3]                 # Q01-20
        ans_roi2 = answer_roi[:, w_ans // 3: 2 * w_ans // 3]   # Q26-45
        ans_roi3 = answer_roi[:, 2 * w_ans // 3: w_ans]        # Q51-70

        h_num, w_num = numeric_roi.shape[:2]
        num_roi1 = numeric_roi[:, 0:w_num // 3]                # Q21-25
        num_roi2 = numeric_roi[:, w_num // 3: 2 * w_num // 3]  # Q46-50
        num_roi3 = numeric_roi[:, 2 * w_num // 3: w_num]       # Q71-75

        # Run detection
        # Use BME logic for roll number detection
        rollno_coords = (0, 0, 485, 224)
        roll_number_raw = self._extract_rollno_digits((cropped_resized, rollno_coords), debug_name="Main Roll Number", is_rollno=True)
        roll_number = roll_number_raw if roll_number_raw != "******" else "ENTER ROLL NO:"

        answers = {}
        # MCQs
        for idx, ans in enumerate(self._detect_answers(ans_roi1), start=1):
            answers[f"Q{idx:02d}"] = ans
        for idx, ans in enumerate(self._detect_answers(ans_roi2), start=26):
            answers[f"Q{idx:02d}"] = ans
        for idx, ans in enumerate(self._detect_answers(ans_roi3), start=51):
            answers[f"Q{idx:02d}"] = ans

        # Numerics
        numeric_rois = [num_roi1, num_roi2, num_roi3]
        roi_start_numbers = [21, 46, 71]
        for i, big_roi in enumerate(numeric_rois, start=1):
            h, w = big_roi.shape[:2]
            step_h = h // 5
            start_q = roi_start_numbers[i - 1]
            for j in range(5):
                qnum = start_q + j
                start_y = j * step_h
                end_y = (j + 1) * step_h if j < 4 else h
                sub_roi = big_roi[start_y:end_y, :]
                raw = self._extract_rollno_digits(sub_roi, debug_name=f"Numeric ROI {i}-{j+1}")
                val = self.parse_numeric_raw_to_float(raw)
                answers[f"Q{qnum:02d}"] = val  # float or None

        return roll_number, answers

# -------------------------------
# Scoring
# -------------------------------
@dataclass
class ScoreBreakdown:
    per_subject: Dict[str, Dict[str, int | float]] = field(default_factory=dict)
    total_correct: int = 0
    total_wrong: int = 0
    total_skipped: int = 0
    total_score: float = 0.0


def normalize_mcq(ans: Any) -> Optional[str]:
    if not isinstance(ans, str):
        return None
    s = ans.strip().upper()
    return s if s in {"A", "B", "C", "D"} else None


def is_numeric_key(key: Any) -> bool:
    return isinstance(key, (int, float))


def fmt_num(val: Optional[float]) -> str:
    if val is None:
        return ""
    s = f"{float(val):.6f}".rstrip('0').rstrip('.')
    return s if s else "0"


def evaluate_sheet(answers: Dict[str, Any], answer_key: Dict[str, Any], scoring_rules: Dict[str, float]) -> ScoreBreakdown:
    per_subject: Dict[str, Dict[str, int | float]] = {}
    total_correct = total_wrong = total_skipped = 0
    total_score = 0.0

    mark_correct = scoring_rules['correct']
    mark_wrong = scoring_rules['wrong']
    mark_skip = scoring_rules['skip']

    for subject, (start, end) in SUBJECTS.items():
        correct = wrong = skipped = 0
        subject_score = 0.0
        for q in range(start, end + 1):
            qk = f"Q{q:02d}"
            det = answers.get(qk, None)
            key = answer_key.get(qk, None)

            if key is None:
                skipped += 1
                continue

            if is_numeric_key(key):
                if isinstance(det, (int, float)):
                    if float(det) == float(key):
                        correct += 1
                        subject_score += mark_correct
                    else:
                        wrong += 1
                        subject_score += mark_wrong
                else:
                    skipped += 1
                    subject_score += mark_skip
            else: # MCQ
                det_norm = normalize_mcq(det)
                key_norm = normalize_mcq(key)
                if key_norm is None:
                    skipped += 1
                    continue
                if det_norm is None:
                    skipped += 1
                    subject_score += mark_skip
                else:
                    if det_norm == key_norm:
                        correct += 1
                        subject_score += mark_correct
                    else:
                        wrong += 1
                        subject_score += mark_wrong

        per_subject[subject] = {
            "correct": correct,
            "wrong": wrong,
            "skipped": skipped,
            "score": subject_score,
            "max": (correct + wrong + skipped) * mark_correct,
        }
        total_correct += correct
        total_wrong += wrong
        total_skipped += skipped
        total_score += subject_score

    return ScoreBreakdown(
        per_subject=per_subject,
        total_correct=total_correct,
        total_wrong=total_wrong,
        total_skipped=total_skipped,
        total_score=total_score,
    )

# -------------------------------
# Data models for UI
# -------------------------------
@dataclass
class StudentResult:
    image_path: str
    roll_number: str
    answers: Dict[str, Any]
    score: ScoreBreakdown


# -------------------------------
# PDF utils
# -------------------------------
def pdf_to_images(pdf_path: str) -> List[str]:
    out_paths: List[str] = []
    if not HAVE_PYMUPDF:
        return out_paths
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            fd, tmp_path = tempfile.mkstemp(prefix=f"omr_{os.path.basename(pdf_path)}_p{i+1}_", suffix=".png")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            out_paths.append(tmp_path)
        doc.close()
    except Exception as e:
        print("PDF conversion failed:", e)
    return out_paths

# -------------------------------
# Worker for background scanning
# -------------------------------
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    sheet_scanned = QtCore.pyqtSignal(object)


class ScanWorker(QtCore.QRunnable):
    def __init__(self, files: List[str], processor: OMRProcessor, answer_key: Dict[str, Any], scoring_rules: Dict[str, float]):
        super().__init__()
        self.files = files
        self.processor = processor
        self.answer_key = answer_key
        self.signals = WorkerSignals()
        self.scoring_rules = scoring_rules

    @QtCore.pyqtSlot()
    def run(self):
        for path in self.files:
            try:
                roll, answers = self.processor.process_image(path)
                score = evaluate_sheet(answers, self.answer_key, self.scoring_rules)
                res = StudentResult(image_path=path, roll_number=roll, answers=answers, score=score)
                self.signals.sheet_scanned.emit(res)
                self.signals.progress.emit(f"Scanned: {os.path.basename(path)} → Roll {roll}")
            except Exception as e:
                self.signals.progress.emit(f"Failed: {os.path.basename(path)} — {e}")
        self.signals.finished.emit()


# -------------------------------
# Matplotlib Canvas for Stats
# -------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

# -------------------------------
# Scoring Rules Dialog
# -------------------------------
class ScoringRulesDialog(QtWidgets.QDialog):
    def __init__(self, parent, current_rules):
        super().__init__(parent)
        self.setWindowTitle("Edit Scoring Rules")
        layout = QtWidgets.QFormLayout(self)

        self.correct_spin = QtWidgets.QDoubleSpinBox()
        self.correct_spin.setRange(-100, 100)
        self.correct_spin.setValue(current_rules['correct'])
        layout.addRow("Marks for Correct:", self.correct_spin)

        self.wrong_spin = QtWidgets.QDoubleSpinBox()
        self.wrong_spin.setRange(-100, 100)
        self.wrong_spin.setValue(current_rules['wrong'])
        layout.addRow("Marks for Wrong:", self.wrong_spin)

        self.skip_spin = QtWidgets.QDoubleSpinBox()
        self.skip_spin.setRange(-100, 100)
        self.skip_spin.setValue(current_rules['skip'])
        layout.addRow("Marks for Skipped:", self.skip_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            "correct": self.correct_spin.value(),
            "wrong": self.wrong_spin.value(),
            "skip": self.skip_spin.value(),
        }

# -------------------------------
# Main Window
# -------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JEE Mains OMR Scanner-BY AZEEM")
        self.resize(1400, 850)

        # State
        self.processor = OMRProcessor()
        self.answer_key: Dict[str, Any] = {f"Q{q:02d}": None for q in range(1, 76)}
        self.results: List[StudentResult] = []
        self.results_map: Dict[str, StudentResult] = {}
        self.thread_pool = QtCore.QThreadPool.globalInstance()

        # Customizable scoring rules
        self.mark_correct = MARK_CORRECT
        self.mark_wrong = MARK_WRONG
        self.mark_skip = MARK_SKIP

        # UI
        self._build_ui()
        self._apply_dark_fusion_theme()

    # ---------- UI setup ----------
    def _build_ui(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(20, 20))
        self.addToolBar(tb)

        actions = [
            ("Load Key (Image/PDF)", self.on_load_key),
            ("Load Sheets (Images/PDF)", self.on_load_sheets),
            ("Scan All", self.on_scan_all),
            ("Export CSV", self.on_export_csv),
            ("Main Export", self.on_export_main),
            ("Scoring Rules", self.on_open_scoring_dialog)
        ]
        for name, slot in actions:
            act = QtGui.QAction(name, self)
            act.triggered.connect(slot)
            tb.addAction(act)

        tb.addSeparator()
        self.chk_debug = QtWidgets.QCheckBox("Debug Windows")
        self.chk_debug.stateChanged.connect(self.on_toggle_debug)
        tb.addWidget(self.chk_debug)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.list_files = QtWidgets.QListWidget()
        self.list_files.itemSelectionChanged.connect(self.on_select_file)
        left_layout.addWidget(QtWidgets.QLabel("Loaded Sheets"))
        left_layout.addWidget(self.list_files, 1)
        self.btn_scan_selected = QtWidgets.QPushButton("Scan Selected")
        self.btn_scan_selected.clicked.connect(self.on_scan_selected)
        left_layout.addWidget(self.btn_scan_selected)
        splitter.addWidget(left_widget)

        center_tabs = QtWidgets.QTabWidget()
        splitter.addWidget(center_tabs)

        # Preview tab
        self.graphics_view = QtWidgets.QGraphicsView()
        self.graphics_view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        center_tabs.addTab(self.graphics_view, "Preview")

        # Answers tab
        self.table_answers = QtWidgets.QTableWidget(75, 4)
        self.table_answers.setHorizontalHeaderLabels(["Q", "Detected", "Key", "Result"])
        self.table_answers.horizontalHeader().setStretchLastSection(True)
        for i in range(75):
            self.table_answers.setItem(i, 0, QtWidgets.QTableWidgetItem(f"Q{i+1:02d}"))
        center_tabs.addTab(self.table_answers, "Per-Question")

        # Results tab
        res_tab = QtWidgets.QWidget()
        res_layout = QtWidgets.QVBoxLayout(res_tab)
        self._build_filter_box(res_layout)
        self.table_results = QtWidgets.QTableWidget(0, 11)
        self.table_results.setHorizontalHeaderLabels([
            "Roll No", "Physics", "Chemistry", "Maths",
            "P C/W/S", "C C/W/S", "M C/W/S", "Total Correct", "Total Wrong", "Total Skipped", "Total Score"
        ])
        self.table_results.setSortingEnabled(True)
        res_layout.addWidget(self.table_results)
        center_tabs.addTab(res_tab, "Results Summary")

        # Statistics Tab
        self._build_statistics_tab(center_tabs)

        # Right panel
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.addWidget(QtWidgets.QLabel("Answer Key"))
        self.table_key = QtWidgets.QTableWidget(75, 2)
        self.table_key.setHorizontalHeaderLabels(["Q", "Key (A-D / number)"])
        self.table_key.horizontalHeader().setStretchLastSection(True)
        for i in range(75):
            self.table_key.setItem(i, 0, QtWidgets.QTableWidgetItem(f"Q{i+1:02d}"))
            self.table_key.setItem(i, 1, QtWidgets.QTableWidgetItem(""))
        right_layout.addWidget(self.table_key)
        self.btn_apply_key = QtWidgets.QPushButton("Apply Key & Rescore")
        self.btn_apply_key.clicked.connect(self.on_apply_key_from_table)
        right_layout.addWidget(self.btn_apply_key)
        splitter.addWidget(right_widget)

        splitter.setSizes([240, 700, 460])
        self.status = self.statusBar()
        self.status.showMessage("Ready.")

    def _build_filter_box(self, parent_layout):
        filter_group = QtWidgets.QGroupBox("Advanced Filtering")
        filter_layout = QtWidgets.QHBoxLayout(filter_group)
        filter_layout.addWidget(QtWidgets.QLabel("Roll No:"))
        self.filter_roll_no = QtWidgets.QLineEdit()
        filter_layout.addWidget(self.filter_roll_no)
        filter_layout.addWidget(QtWidgets.QLabel("Min Score:"))
        self.filter_min_score = QtWidgets.QDoubleSpinBox()
        self.filter_min_score.setRange(-TOTAL_MAX * 2, TOTAL_MAX * 2)
        self.filter_min_score.setValue(-TOTAL_MAX * 2)
        filter_layout.addWidget(self.filter_min_score)
        filter_layout.addWidget(QtWidgets.QLabel("Max Score:"))
        self.filter_max_score = QtWidgets.QDoubleSpinBox()
        self.filter_max_score.setRange(-TOTAL_MAX * 2, TOTAL_MAX * 2)
        self.filter_max_score.setValue(TOTAL_MAX * 2)
        filter_layout.addWidget(self.filter_max_score)
        self.btn_clear_filter = QtWidgets.QPushButton("Clear")
        filter_layout.addWidget(self.btn_clear_filter)
        parent_layout.addWidget(filter_group)

        self.filter_roll_no.textChanged.connect(self._apply_results_filter)
        self.filter_min_score.valueChanged.connect(self._apply_results_filter)
        self.filter_max_score.valueChanged.connect(self._apply_results_filter)
        self.btn_clear_filter.clicked.connect(self._clear_filters)

    def _build_statistics_tab(self, center_tabs):
        stats_tab = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_tab)
        if HAVE_MATPLOTLIB:
            self.stats_canvas = MplCanvas(self, width=8, height=6, dpi=100)
            stats_layout.addWidget(self.stats_canvas)
            btn_refresh_stats = QtWidgets.QPushButton("Refresh Statistics")
            btn_refresh_stats.clicked.connect(self._update_statistics_dashboard)
            stats_layout.addWidget(btn_refresh_stats)
        else:
            label = QtWidgets.QLabel("Statistics dashboard requires 'matplotlib'.\nPlease install it using: pip install matplotlib")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stats_layout.addWidget(label)
        center_tabs.addTab(stats_tab, "Statistics")
        if not HAVE_MATPLOTLIB:
            center_tabs.setTabEnabled(center_tabs.indexOf(stats_tab), False)

    def _apply_dark_fusion_theme(self):
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(220, 0, 0))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(0, 0, 0))
        app.setPalette(palette)

    # ---------- Actions and Slots ----------
    def on_toggle_debug(self, state: int):
        self.processor.SHOW_DEBUG = (state == Qt.CheckState.Checked.value)

    def on_load_key(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Key Image/PDF", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)")
        if not paths: return
        key_answers: Dict[str, Any] = {f"Q{q:02d}": None for q in range(1, 76)}
        for path in paths:
            images = [path] if not path.lower().endswith(".pdf") else pdf_to_images(path)
            for img in images:
                try:
                    _, ans = self.processor.process_image(img)
                    for qk, val in ans.items():
                        if val is not None and val not in ("null", "X"):
                            key_answers[qk] = val
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Key Scan Failed", f"{os.path.basename(img)}: {e}")
        for i in range(75):
            val = key_answers.get(f"Q{i+1:02d}")
            txt = fmt_num(val) if isinstance(val, (int, float)) else (val or "")
            self.table_key.item(i, 1).setText(txt)
        self.on_apply_key_from_table()
        self.status.showMessage("Key loaded and applied. Scores recalculated.")

    def on_load_sheets(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select OMR Sheets", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)")
        if not paths: return
        for path in paths:
            if path.lower().endswith(".pdf"):
                self.list_files.addItems(pdf_to_images(path))
            else:
                self.list_files.addItem(path)
        self.status.showMessage(f"Loaded {self.list_files.count()} sheet image(s).")

    def _start_scan(self, files: List[str]):
        scoring_rules = {"correct": self.mark_correct, "wrong": self.mark_wrong, "skip": self.mark_skip}
        worker = ScanWorker(files, self.processor, self.answer_key, scoring_rules)
        worker.signals.finished.connect(lambda: self.status.showMessage("Scan complete."))
        worker.signals.progress.connect(self.status.showMessage)
        worker.signals.sheet_scanned.connect(self.on_sheet_scanned)
        self.thread_pool.start(worker)
        self.status.showMessage("Scanning…")

    def on_scan_selected(self):
        files = [i.text() for i in self.list_files.selectedItems()]
        if files: self._start_scan(files)

    def on_scan_all(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        if files: self._start_scan(files)

    def on_sheet_scanned(self, result: StudentResult):
        if result.image_path in self.results_map:
            for idx, r in enumerate(self.results):
                if r.image_path == result.image_path:
                    self.results[idx] = result; break
        else:
            self.results.append(result)
        self.results_map[result.image_path] = result
        self.results.sort(key=lambda r: r.roll_number)
        self._refresh_results_table()
        self._apply_results_filter()
        if self.list_files.currentItem() and self.list_files.currentItem().text() == result.image_path:
            self._show_preview(result)

    def on_select_file(self):
        item = self.list_files.currentItem()
        if not item: return
        path = item.text()
        res = self.results_map.get(path)
        if res:
            self._show_preview(res)
            # Ensure the table_answers items are created if they don't exist
            self._ensure_table_answers_items()
        else:
            self._show_image(path)
            for i in range(75): # Clear old data
                self.table_answers.setItem(i, 1, QtWidgets.QTableWidgetItem(""))
                self.table_answers.setItem(i, 3, QtWidgets.QTableWidgetItem(""))

    def on_apply_key_from_table(self):
        for i in range(75):
            qk = f"Q{i+1:02d}"
            txt = self.table_key.item(i, 1).text().strip()
            if not txt: self.answer_key[qk] = None; continue
            if txt.upper() in "ABCD": self.answer_key[qk] = txt.upper(); continue
            try: self.answer_key[qk] = float(txt)
            except ValueError: self.answer_key[qk] = None
        self.status.showMessage("Answer key updated. Recalculating all scores...")
        self._recalculate_all_scores()

    def on_open_scoring_dialog(self):
        current = {"correct": self.mark_correct, "wrong": self.mark_wrong, "skip": self.mark_skip}
        dialog = ScoringRulesDialog(self, current)
        if dialog.exec():
            new = dialog.get_values()
            self.mark_correct, self.mark_wrong, self.mark_skip = new['correct'], new['wrong'], new['skip']
            self.status.showMessage("Scoring rules updated. Recalculating scores...")
            self._recalculate_all_scores()

    def _recalculate_all_scores(self):
        scoring_rules = {"correct": self.mark_correct, "wrong": self.mark_wrong, "skip": self.mark_skip}
        for res in self.results:
            res.score = evaluate_sheet(res.answers, self.answer_key, scoring_rules)
        self._refresh_results_table()
        self._apply_results_filter()
        self.on_select_file() # Refresh detailed view if an item is selected
        self.status.showMessage("All scores recalculated.")

    def on_export_csv(self):
        # Ensure all results are up-to-date before exporting
        self._recalculate_all_scores()
        # Ensure the table_answers items are created if they don't exist
        self._ensure_table_answers_items()

        if not self.results: return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "results.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["Image", "RollNumber", "PhysicsScore", "ChemistryScore", "MathsScore",
                          "Physics_C/W/S", "Chemistry_C/W/S", "Maths_C/W/S",
                          "TotalCorrect", "TotalWrong", "TotalSkipped", "TotalScore"]
                header += [f"Q{q:02d}" for q in range(1, 76)]
                writer.writerow(header)
                for res in self.results:
                    p, c, m = res.score.per_subject.get("Physics", {}), res.score.per_subject.get("Chemistry", {}), res.score.per_subject.get("Maths", {})
                    row = [res.image_path, res.roll_number,
                           p.get("score", 0), c.get("score", 0), m.get("score", 0),
                           f"{p.get('correct',0)}/{p.get('wrong',0)}/{p.get('skipped',0)}",
                           f"{c.get('correct',0)}/{c.get('wrong',0)}/{c.get('skipped',0)}",
                           f"{m.get('correct',0)}/{m.get('wrong',0)}/{m.get('skipped',0)}",
                           res.score.total_correct, res.score.total_wrong, res.score.total_skipped, res.score.total_score]
                    for q in range(1, 76):
                        val = res.answers.get(f"Q{q:02d}")
                        row.append(fmt_num(val) if isinstance(val, (int, float)) else (val if str(val).upper() in "ABCD" else ""))
                    writer.writerow(row)
            self.status.showMessage(f"Exported CSV → {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Failed", str(e))

    def on_export_main(self):
        """Export a ranked summary CSV with columns: Rank, RollNo, Physics, Chemistry, Maths, TotalScore"""
        if not self.results:
            QtWidgets.QMessageBox.information(self, "Main Export", "No results to export. Scan some sheets first.")
            return
        # Recalculate to be safe
        self._recalculate_all_scores()
        # Sort by total_score desc, then roll no
        ranked = sorted(self.results, key=lambda r: (-r.score.total_score, r.roll_number))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Main Export CSV", "main_export.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Rank", "RollNo", "Physics", "Chemistry", "Maths", "TotalScore"])
                for rank, res in enumerate(ranked, start=1):
                    p = res.score.per_subject.get("Physics", {}).get("score", 0)
                    c = res.score.per_subject.get("Chemistry", {}).get("score", 0)
                    m = res.score.per_subject.get("Maths", {}).get("score", 0)
                    writer.writerow([rank, res.roll_number, p, c, m, res.score.total_score])
            self.status.showMessage(f"Main export written → {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Failed", str(e))

    # ---------- UI Update methods ----------
    def _show_image(self, path: str):
        pix = QtGui.QPixmap(path)
        self.scene.clear()
        self.scene.addPixmap(pix)
        self.graphics_view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _ensure_table_answers_items(self):
        # This method ensures that QTableWidgetItem objects exist in the table_answers
        # before attempting to set their text.
        for i in range(self.table_answers.rowCount()):
            for j in range(self.table_answers.columnCount()):
                if self.table_answers.item(i, j) is None:
                    self.table_answers.setItem(i, j, QtWidgets.QTableWidgetItem(""))

    def _show_preview(self, res: StudentResult):
        self._show_image(res.image_path)
        for i in range(75):
            qk = f"Q{i+1:02d}"
            det, key = res.answers.get(qk), self.answer_key.get(qk)
            det_txt = fmt_num(det) if isinstance(det, (int, float)) else (det if str(det).upper() in "ABCD" else "")
            key_txt = fmt_num(key) if isinstance(key, (int, float)) else (str(key) if key is not None else "")

            # Ensure items exist before setting text
            if self.table_answers.item(i, 1) is None:
                self.table_answers.setItem(i, 1, QtWidgets.QTableWidgetItem(""))
            self.table_answers.item(i, 1).setText(det_txt)
            if self.table_answers.item(i, 2) is None:
                self.table_answers.setItem(i, 2, QtWidgets.QTableWidgetItem(""))
            self.table_answers.item(i, 2).setText(key_txt)

            result_txt = ""
            if key is not None:
                if is_numeric_key(key):
                    result_txt = "Correct" if isinstance(det, (int, float)) and float(det) == float(key) else ("Wrong" if isinstance(det, (int, float)) else "Skipped")
                else:
                    result_txt = "Correct" if normalize_mcq(det) == normalize_mcq(key) else ("Wrong" if normalize_mcq(det) else "Skipped")
            
            it = QtWidgets.QTableWidgetItem(result_txt)
            if result_txt == "Correct": it.setForeground(QtGui.QColor(0, 200, 0))
            elif result_txt == "Wrong": it.setForeground(QtGui.QColor(220, 0, 0))
            else: it.setForeground(QtGui.QColor(200, 200, 120))
            self.table_answers.setItem(i, 3, it)

    def _refresh_results_table(self):
        self.table_results.setSortingEnabled(False)
        self.table_results.setRowCount(len(self.results))
        for r, res in enumerate(self.results):
            roll_item = QtWidgets.QTableWidgetItem(str(res.roll_number))
            roll_item.setData(Qt.ItemDataRole.UserRole, res.image_path)
            self.table_results.setItem(r, 0, roll_item)
            
            p, c, m = res.score.per_subject.get("Physics", {}), res.score.per_subject.get("Chemistry", {}), res.score.per_subject.get("Maths", {})
            data = [
                str(p.get("score", 0)), str(c.get("score", 0)), str(m.get("score", 0)),
                f"{p.get('correct',0)}/{p.get('wrong',0)}/{p.get('skipped',0)}",
                f"{c.get('correct',0)}/{c.get('wrong',0)}/{c.get('skipped',0)}",
                f"{m.get('correct',0)}/{m.get('wrong',0)}/{m.get('skipped',0)}",
                str(res.score.total_correct), str(res.score.total_wrong), str(res.score.total_skipped), str(res.score.total_score)
            ]
            for c_idx, text in enumerate(data, 1):
                self.table_results.setItem(r, c_idx, QtWidgets.QTableWidgetItem(text))
        self.table_results.resizeColumnsToContents()
        self.table_results.setSortingEnabled(True)

    def _clear_filters(self):
        self.filter_roll_no.setText("")
        self.filter_min_score.setValue(self.filter_min_score.minimum())
        self.filter_max_score.setValue(self.filter_max_score.maximum())

    def _apply_results_filter(self):
        roll_filter, min_score, max_score = self.filter_roll_no.text().lower(), self.filter_min_score.value(), self.filter_max_score.value()
        for r in range(self.table_results.rowCount()):
            res = self.results_map.get(self.table_results.item(r, 0).data(Qt.ItemDataRole.UserRole))
            if not res: continue
            roll_match = roll_filter in res.roll_number.lower()
            score_match = min_score <= res.score.total_score <= max_score
            self.table_results.setRowHidden(r, not (roll_match and score_match))

    def _get_filtered_results(self) -> List[StudentResult]:
        return [self.results_map[self.table_results.item(r, 0).data(Qt.ItemDataRole.UserRole)]
                for r in range(self.table_results.rowCount()) if not self.table_results.isRowHidden(r)]

    def _update_statistics_dashboard(self):
        if not HAVE_MATPLOTLIB: return
        results = self._get_filtered_results()
        self.stats_canvas.fig.clear()
        if not results:
            ax = self.stats_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display.", ha='center', va='center')
            self.stats_canvas.draw()
            return

        total_scores = [r.score.total_score for r in results]
        avg_c = np.mean([r.score.total_correct for r in results])
        avg_w = np.mean([r.score.total_wrong for r in results])
        avg_s = np.mean([r.score.total_skipped for r in results])
        avg_subj = {s: np.mean([r.score.per_subject[s]['score'] for r in results]) for s in SUBJECTS}

        self.stats_canvas.fig.suptitle("Performance Statistics (Filtered Results)")
        ax1 = self.stats_canvas.fig.add_subplot(2, 2, 1)
        ax1.hist(total_scores, bins=15, edgecolor='black')
        ax1.set_title("Total Score Distribution"); ax1.set_xlabel("Score"); ax1.set_ylabel("Students")

        ax2 = self.stats_canvas.fig.add_subplot(2, 2, 2)
        ax2.pie([avg_c, avg_w, avg_s], labels=[f"Correct\n({avg_c:.1f})", f"Wrong\n({avg_w:.1f})", f"Skipped\n({avg_s:.1f})"],
                autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'])
        ax2.set_title("Average Answer Breakdown"); ax2.axis('equal')

        ax3 = self.stats_canvas.fig.add_subplot(2, 1, 2)
        ax3.bar(avg_subj.keys(), avg_subj.values(), color=['#007ACC', '#D4AC0D', '#900C3F'])
        ax3.set_title("Average Score by Subject"); ax3.set_ylabel("Average Score")

        self.stats_canvas.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.stats_canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    class LicenseDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("License Verification")
            self.setModal(True)
            self.setFixedSize(380, 140)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel("Enter license key to continue:"))

            self.input = QtWidgets.QLineEdit()
            self.input.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
            layout.addWidget(self.input)

            btn_layout = QtWidgets.QHBoxLayout()
            self.btn_verify = QtWidgets.QPushButton("Verify")
            self.btn_cancel = QtWidgets.QPushButton("Exit")
            btn_layout.addStretch(1)
            btn_layout.addWidget(self.btn_verify)
            btn_layout.addWidget(self.btn_cancel)
            layout.addLayout(btn_layout)

            self.btn_verify.clicked.connect(self.on_verify)
            self.btn_cancel.clicked.connect(self.reject)

            self.verified_payload = None

        def on_verify(self):
            key = self.input.text().strip()
            if not key:
                QtWidgets.QMessageBox.warning(self, "Invalid", "Please enter a license key.")
                return
            ok, payload_or_msg = verify_license_with_supabase(key)
            if ok:
                self.verified_payload = payload_or_msg
                QtWidgets.QMessageBox.information(self, "Verified", "License key verified. Thank you.")
                self.accept()
            else:
                QtWidgets.QMessageBox.critical(self, "Verification Failed", str(payload_or_msg))

    def verify_license_with_supabase(key: str, timeout=6):
        """Verify a license key against Supabase REST. Expects two env vars:
        SUPABASE_URL (e.g. https://xyz.supabase.co) and SUPABASE_KEY (anon/public key).
        Returns (True, payload) on success, (False, error_message) on failure.
        """
        # Allow env override; fall back to the provided project values
        supabase_url = os.environ.get("SUPABASE_URL", "https://ymqlzkdysieexusqzllx.supabase.co")
        supabase_key = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InltcWx6a2R5c2llZXh1c3F6bGx4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3MDc3OTksImV4cCI6MjA3MTI4Mzc5OX0.5iWccnZY_gsT9vv9205yf09JBj-IUG2-hyxYwmTQUpk")
        if os.environ.get("SUPABASE_DISABLE") == "1":
            return True, {"note": "License checks disabled via SUPABASE_DISABLE=1"}
        if not supabase_url or not supabase_key:
            return False, "SUPABASE_URL or SUPABASE_KEY not set in environment."

        rest_endpoint = supabase_url.rstrip("/") + "/rest/v1/licenses"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Accept": "application/json",
        }
        params = {
            "select": "*",
            "license_key": f"eq.{key}"
        }
        try:
            r = requests.get(rest_endpoint, headers=headers, params=params, timeout=timeout)
        except Exception as e:
            return False, f"Network error: {e}"
        if r.status_code != 200:
            return False, f"Supabase returned status {r.status_code}: {r.text}"
        try:
            data = r.json()
        except Exception:
            return False, "Failed to parse response from Supabase."
        if not data:
            return False, "License key not found."
        # Expecting a row with an 'active' boolean column
        row = data[0]
        if row.get("active") is True:
            return True, row
        else:
            return False, f"License inactive or revoked (active={row.get('active')})."

    dlg = LicenseDialog()
    # Simple cache: avoid re-verifying if a valid key was recently verified
    cache_path = os.path.join(tempfile.gettempdir(), "omr_license_cache.json")
    def load_cache():
        try:
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_cache(key, payload):
        try:
            obj = {"key": key, "payload": payload, "verified_at": datetime.now(timezone.utc).isoformat()}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        except Exception:
            pass

    cached = load_cache()
    auto_accepted = False
    # If we have a cached key, always re-verify it against Supabase before auto-accepting
    if cached:
        try:
            ckey = cached.get("key")
            if ckey:
                ok, payload_or_msg = verify_license_with_supabase(ckey)
                if ok:
                    # remote verification success -> refresh cache and accept
                    save_cache(ckey, payload_or_msg)
                    auto_accepted = True
                else:
                    # remote verification failed (inactive/expired/etc.) -> fall through to prompt
                    auto_accepted = False
        except Exception:
            # any unexpected error -> do not auto-accept, show dialog
            auto_accepted = False

    if not auto_accepted:
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            # user cancelled or verification failed
            sys.exit(1)
        # save cache for the newly verified key
        if dlg.verified_payload:
            save_cache(dlg.input.text().strip(), dlg.verified_payload)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
