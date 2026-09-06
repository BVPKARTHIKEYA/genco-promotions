import os
import re
import sys
import csv
import glob
import time
import hashlib
import pickle

import numpy as np
import cv2
import pymupdf
import pytesseract

from PIL import Image

from dotenv import load_dotenv
import chromadb
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PDF_DIR = os.path.join(
    BASE_DIR,
    "knowledge-base"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

PICKLE_FILE = os.path.join(
    BASE_DIR,
    "chunks.pkl"
)

CSV_DIR = os.path.join(
    BASE_DIR,
    "extracted_tables"
)

COLLECTION_NAME = "knowledge_base"


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


# ============================================================
# TABLE COLUMNS
# ============================================================

COLUMN_LABELS = [

    "S.No",

    "Name of the employee",

    "Present place of working",

    "Emp ID.No",

    "Reference of Promotion order",

    "Now Revised date of promotion"
]

NAME_COLUMN = "Name of the employee"

ID_COLUMN = "Emp ID.No"

PLACE_COLUMN = "Present place of working"

REFERENCE_COLUMN = "Reference of Promotion order"

REVISED_DATE_COLUMN = "Now Revised date of promotion"


# ============================================================
# OCR SETTINGS
# ============================================================

RENDER_ZOOM = 3.0

OCR_MIN_CONFIDENCE = 15

ROW_Y_TOLERANCE = 22

BODY_ROW_LEFT_MAX = 450


# ============================================================
# EMBEDDING
# ============================================================

EMBED_MODEL = "gemini-embedding-001"

BATCH_SIZE = 10

BATCH_DELAY = 3


# ============================================================
# NARRATIVE CHUNKING
# ============================================================

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200


# ============================================================
# GEMINI CLIENT
# ============================================================

def configure_gemini():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        print(
            "\nERROR: GEMINI_API_KEY not found."
        )

        print(
            "\nAdd this to .env:"
        )

        print(
            "GEMINI_API_KEY=your_api_key_here"
        )

        sys.exit(1)

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# CHECK TESSERACT
# ============================================================

def check_tesseract():

    print(
        "\nChecking Tesseract OCR..."
    )

    print(
        f"Tesseract path: {TESSERACT_PATH}"
    )

    if not os.path.exists(
        TESSERACT_PATH
    ):

        print(
            "\nERROR: Tesseract executable was not found."
        )

        print(
            TESSERACT_PATH
        )

        return False

    try:

        version = (
            pytesseract
            .get_tesseract_version()
        )

        print(
            f"Tesseract OCR detected: {version}"
        )

        return True

    except Exception as e:

        print(
            "\nERROR: Tesseract OCR could not be started."
        )

        print(
            f"Details: {e}"
        )

        return False


# ============================================================
# RENDER PAGE
# ============================================================

def render_page_image(
    page,
    zoom=RENDER_ZOOM
):

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(
            zoom,
            zoom
        ),
        alpha=False
    )

    return np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        3
    )


# ============================================================
# REMOVE GRIDLINES FOR OCR
# ============================================================

def remove_gridlines(
    rgb_image
):

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )[1]

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (50, 1)
    )

    h_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        h_kernel,
        iterations=2
    )

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 50)
    )

    v_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        v_kernel,
        iterations=2
    )

    lines_mask = cv2.bitwise_or(
        h_lines,
        v_lines
    )

    cleaned = cv2.subtract(
        binary,
        lines_mask
    )

    return Image.fromarray(
        cv2.bitwise_not(
            cleaned
        )
    )


# ============================================================
# OCR WORDS
# ============================================================

_NOISE_TOKEN_RE = re.compile(
    r'^[|_\[\]\\/~`"\'.,;:]+$'
)


def ocr_words(
    pil_image
):

    data = pytesseract.image_to_data(
        pil_image,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )

    data = data[
        (data.conf > OCR_MIN_CONFIDENCE)
        &
        (data.text.notna())
    ]

    if data.empty:

        return []

    data = data[
        data.text
        .astype(str)
        .str.strip()
        != ""
    ]

    words = []

    for _, row in data.iterrows():

        text = str(
            row["text"]
        ).strip()

        if not text:

            continue

        if _NOISE_TOKEN_RE.match(
            text
        ):

            continue

        words.append(
            {
                "left": float(
                    row["left"]
                ),

                "top": float(
                    row["top"]
                ),

                "width": float(
                    row["width"]
                ),

                "height": float(
                    row["height"]
                ),

                "text": text
            }
        )

    return words


# ============================================================
# PLAIN OCR
# ============================================================

def ocr_plain_text(
    pil_image
):

    return pytesseract.image_to_string(
        pil_image,
        config="--oem 3 --psm 6"
    ).strip()


# ============================================================
# GROUP WORDS INTO ROWS
# ============================================================

def group_into_rows(
    words,
    row_tol=ROW_Y_TOLERANCE
):

    words = sorted(
        words,
        key=lambda w: (
            w["top"],
            w["left"]
        )
    )

    rows = []

    current = []

    current_mean = None

    for word in words:

        y = word["top"]

        if (
            current
            and
            abs(
                y
                -
                current_mean
            )
            >
            row_tol
        ):

            rows.append(
                current
            )

            current = []

        current.append(
            word
        )

        current_mean = (
            sum(
                x["top"]
                for x in current
            )
            /
            len(current)
        )

    if current:

        rows.append(
            current
        )

    for row in rows:

        row.sort(
            key=lambda w: w["left"]
        )

    return rows


# ============================================================
# CHECK BODY ROW
# ============================================================

def looks_like_body_row(
    row
):

    if not row:

        return False

    first_text = str(
        row[0]["text"]
    ).strip()

    # Normal S.No
    digits = re.sub(
        r"\D",
        "",
        first_text
    )

    return (
        digits.isdigit()
        and
        1 <= len(digits) <= 4
        and
        row[0]["left"]
        <
        BODY_ROW_LEFT_MAX
    )


# ============================================================
# CLEAN VALUE
# ============================================================

def clean_value(
    value
):

    if value is None:

        return ""

    value = str(
        value
    )

    value = value.replace(
        "\n",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip()

    # Remove OCR quotes around a complete value
    value = re.sub(
        r"^[\"'“‘`]+",
        "",
        value
    )

    value = re.sub(
        r"[\"'”’`]+$",
        "",
        value
    )

    return value.strip(
        " :;-|\t*_"
    ).strip(
        "\"'“”‘’`"
    ).strip()


# ============================================================
# EMPLOYEE ID
# ============================================================

_EMP_ID_RE = re.compile(
    r"\b\d{6,8}\b"
)


# ============================================================
# VALID EMPLOYEE NAME
# ============================================================

def is_valid_employee_name(
    name
):

    name = clean_value(
        name
    )

    if not name:

        return False

    normalized = name.lower()

    bad_phrases = [

        "through",
        "copy to",
        "copy",
        "controlling officer",
        "controlling officers",
        "chief general manager",
        "general manager",
        "manager",
        "officer",
        "officers",
        "vidyut soudha",
        "vijayawada",
        "appdcl",
        "address",
        "reference",
        "promotion order",
        "revised date",
        "present place",
        "place of working",
        "employee id",
        "emp id",
        "subject",
        "signature",
        "the chief",
        "copies to"
    ]

    for phrase in bad_phrases:

        if phrase in normalized:

            return False

    if len(name) > 80:

        return False

    if len(name.split()) > 7:

        return False

    punctuation_count = len(
        re.findall(
            r"[,;:/|]",
            name
        )
    )

    if punctuation_count >= 3:

        return False

    if not re.search(
        r"[A-Za-z]",
        name
    ):

        return False

    return True


# ============================================================
# FIND TABLE VERTICAL LINES
# ============================================================

def detect_vertical_table_lines(
    rgb_image
):

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        +
        cv2.THRESH_OTSU
    )[1]

    h, w = gray.shape

    kernel_height = max(
        80,
        int(h * 0.10)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, kernel_height)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    projection = (
        np.sum(
            vertical > 0,
            axis=0
        )
    )

    threshold = max(
        50,
        int(h * 0.15)
    )

    candidate_x = np.where(
        projection > threshold
    )[0]

    if len(candidate_x) == 0:

        return []

    groups = []

    current = [
        int(candidate_x[0])
    ]

    for x in candidate_x[1:]:

        x = int(x)

        if (
            x
            -
            current[-1]
            <= 5
        ):

            current.append(
                x
            )

        else:

            groups.append(
                current
            )

            current = [
                x
            ]

    groups.append(
        current
    )

    centers = [

        int(
            round(
                sum(group)
                /
                len(group)
            )
        )

        for group in groups

        if len(group) >= 1
    ]

    # Merge very close detections
    merged = []

    for x in centers:

        if not merged:

            merged.append(x)

        elif (
            x
            -
            merged[-1]
            <
            20
        ):

            merged[-1] = int(
                (
                    merged[-1]
                    +
                    x
                )
                /
                2
            )

        else:

            merged.append(x)

    return merged


# ============================================================
# FALLBACK COLUMN BOUNDARIES
# ============================================================

def fallback_column_boundaries(
    body_rows,
    image_width
):

    # Try to estimate the seven vertical boundaries from
    # word positions when grid detection is unavailable.

    starts = []

    for row in body_rows:

        for word in row:

            starts.append(
                word["left"]
            )

    if not starts:

        return []

    # Expected approximate proportions for the six-column
    # promotion table.
    proportions = [

        0.00,
        0.065,
        0.335,
        0.525,
        0.635,
        0.885,
        1.00
    ]

    return [
        int(
            p * image_width
        )
        for p in proportions
    ]


# ============================================================
# CHOOSE TABLE BOUNDARIES
# ============================================================

def get_column_boundaries(
    rgb_image,
    body_rows
):

    height, width = rgb_image.shape[:2]

    detected = detect_vertical_table_lines(
        rgb_image
    )

    # We need seven boundaries for six columns.
    if len(detected) >= 7:

        # Prefer detections that span a meaningful portion
        # of the page and cover the body/table region.
        detected = sorted(
            detected
        )

        # Reduce excessive vertical lines by keeping
        # sufficiently separated lines.
        filtered = []

        for x in detected:

            if not filtered:

                filtered.append(x)

            elif (
                x
                -
                filtered[-1]
                >=
                width * 0.02
            ):

                filtered.append(x)

        if len(filtered) >= 7:

            # The table is expected to have exactly seven
            # useful vertical boundaries.
            #
            # Select the seven lines that best distribute
            # across the page.
            if len(filtered) > 7:

                target_positions = np.linspace(
                    filtered[0],
                    filtered[-1],
                    7
                )

                selected = []

                for target in target_positions:

                    nearest = min(
                        filtered,
                        key=lambda x:
                        abs(x - target)
                    )

                    if (
                        not selected
                        or
                        nearest
                        !=
                        selected[-1]
                    ):

                        selected.append(
                            nearest
                        )

                if len(selected) == 7:

                    return selected

            return filtered[:7]

    return fallback_column_boundaries(
        body_rows,
        width
    )


# ============================================================
# ASSIGN WORD TO COLUMN
# ============================================================

def word_center(
    word
):

    return (
        word["left"]
        +
        (
            word.get(
                "width",
                0
            )
            /
            2
        )
    )


def assign_word_to_column(
    word,
    boundaries
):

    center = word_center(
        word
    )

    for i in range(
        len(boundaries) - 1
    ):

        left = boundaries[i]

        right = boundaries[i + 1]

        if (
            left
            <=
            center
            <
            right
        ):

            return i

    if center < boundaries[0]:

        return 0

    return len(boundaries) - 2


# ============================================================
# RECONSTRUCT TABLE
# ============================================================

def reconstruct_table(
    rows,
    rgb_image
):

    body_rows = [

        row

        for row in rows

        if looks_like_body_row(
            row
        )
    ]

    if not body_rows:

        return []

    boundaries = get_column_boundaries(
        rgb_image,
        body_rows
    )

    if len(boundaries) != 7:

        print(
            "  WARNING: Could not determine "
            "six table columns."
        )

        return []

    table_rows = []

    for row in body_rows:

        cells = [

            []

            for _ in range(6)
        ]

        # ----------------------------------------------------
        # Assign every OCR word to exactly one cell
        # ----------------------------------------------------

        for word in row:

            column_index = assign_word_to_column(
                word,
                boundaries
            )

            if 0 <= column_index < 6:

                cells[column_index].append(
                    word
                )

        # ----------------------------------------------------
        # Convert cells to text
        # ----------------------------------------------------

        values = []

        for cell_words in cells:

            cell_words = sorted(
                cell_words,
                key=lambda w: (
                    w["top"],
                    w["left"]
                )
            )

            value = clean_value(
                " ".join(
                    word["text"]
                    for word in cell_words
                )
            )

            values.append(
                value
            )

        # ----------------------------------------------------
        # Empty S.No / bad OCR
        # ----------------------------------------------------

        serial_number = values[0]

        if not re.sub(
            r"\D",
            "",
            serial_number
        ):

            continue

        # ----------------------------------------------------
        # Employee name
        # ----------------------------------------------------

        employee_name = clean_value(
            values[1]
        )

        if not is_valid_employee_name(
            employee_name
        ):

            continue

        # ----------------------------------------------------
        # Employee ID
        # ----------------------------------------------------

        employee_id_raw = values[3]

        id_match = _EMP_ID_RE.search(
            employee_id_raw
        )

        employee_id = (

            id_match.group(0)

            if id_match

            else ""
        )

        # A real employee row should have an employee ID.
        if not employee_id:

            # Try the complete row
            full_row_text = " ".join(
                values
            )

            id_match = _EMP_ID_RE.search(
                full_row_text
            )

            if id_match:

                employee_id = (
                    id_match.group(0)
                )

        if not employee_id:

            continue

        # ----------------------------------------------------
        # Build record
        # ----------------------------------------------------

        record = {

            "S.No":
                clean_value(
                    serial_number
                ),

            "Name of the employee":
                employee_name,

            "Present place of working":
                clean_value(
                    values[2]
                ),

            "Emp ID.No":
                employee_id,

            "Reference of Promotion order":
                clean_value(
                    values[4]
                ),

            "Now Revised date of promotion":
                clean_value(
                    values[5]
                )
        }

        table_rows.append(
            record
        )

    return table_rows


# ============================================================
# FIND G.O. REFERENCE
# ============================================================

_GOO_REF_RE = re.compile(
    r"(?:G\.?O\.?\.?|GO)"
    r"\s*No\.?"
    r"\s*[\w./-]+"
    r".{0,60}?"
    r"(?:Dt|Date)"
    r"[:.]?"
    r"\s*[\d./-]+",
    re.IGNORECASE
)


def find_goo_reference(
    plain_text
):

    match = _GOO_REF_RE.search(
        plain_text
    )

    if match:

        return clean_value(
            match.group(0)
        )

    return "unknown reference"


# ============================================================
# ROW TO CHUNK TEXT
# ============================================================

def row_to_chunk_text(
    pdf_name,
    page_number,
    goo_reference,
    record
):

    lines = [

        (
            f"Employee promotion record - "
            f"Source: {pdf_name}, "
            f"Page {page_number}"
        ),

        (
            f"Order reference: "
            f"{goo_reference}"
        )
    ]

    for label in COLUMN_LABELS:

        value = clean_value(
            record.get(
                label,
                ""
            )
        )

        if value:

            lines.append(
                f"{label}: {value}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# CREATE PICKLE RECORD
# ============================================================

def create_pickle_record(
    pdf_name,
    page_number,
    record,
    chunk_text_value,
    goo_reference
):

    name_value = clean_value(
        record.get(
            NAME_COLUMN,
            ""
        )
    )

    id_value = clean_value(
        record.get(
            ID_COLUMN,
            ""
        )
    )

    place_value = clean_value(
        record.get(
            PLACE_COLUMN,
            ""
        )
    )

    reference_value = clean_value(
        record.get(
            REFERENCE_COLUMN,
            ""
        )
    )

    revised_date_value = clean_value(
        record.get(
            REVISED_DATE_COLUMN,
            ""
        )
    )

    return {

        "text":
            chunk_text_value,

        "source":
            pdf_name,

        "page":
            page_number,

        "row_type":
            "table_row",

        "name":
            name_value,

        "name_lower":
            name_value.lower(),

        "emp_id":
            id_value,

        "employee_name":
            name_value,

        "employee_id":
            id_value,

        "goo_reference":
            goo_reference,

        "s_no":
            clean_value(
                record.get(
                    "S.No",
                    ""
                )
            ),

        "present_place":
            place_value,

        "present_place_of_working":
            place_value,

        "reference":
            reference_value,

        "promotion_reference":
            reference_value,

        "reference_of_promotion_order":
            reference_value,

        "revised_date":
            revised_date_value,

        "now_revised_date":
            revised_date_value,

        "now_revised_date_of_promotion":
            revised_date_value
    }


# ============================================================
# NARRATIVE CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    n = len(
        text
    )

    while start < n:

        end = min(
            start
            +
            chunk_size,
            n
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end == n:

            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def embed_texts(
    client,
    texts
):

    if not texts:

        return []

    max_retries = 5

    for attempt in range(
        max_retries
    ):

        try:

            result = (
                client.models.embed_content(

                    model=EMBED_MODEL,

                    contents=texts,

                    config={
                        "task_type":
                        "RETRIEVAL_DOCUMENT"
                    }
                )
            )

            embeddings = [

                embedding.values

                for embedding
                in result.embeddings
            ]

            if len(embeddings) != len(
                texts
            ):

                raise RuntimeError(
                    f"Got "
                    f"{len(embeddings)} "
                    f"embeddings for "
                    f"{len(texts)} texts."
                )

            return embeddings

        except Exception as e:

            message = str(
                e
            )

            rate_limited = (

                "429" in message

                or

                "RESOURCE_EXHAUSTED"
                in
                message

                or

                "quota"
                in
                message.lower()
            )

            if (
                rate_limited
                and
                attempt
                <
                max_retries - 1
            ):

                wait_seconds = (
                    10
                    *
                    (
                        2
                        ** attempt
                    )
                )

                print(
                    f"  Rate limited, "
                    f"waiting "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            print(
                f"  ERROR creating "
                f"embeddings: {e}"
            )

            raise

    return []


# ============================================================
# CHUNK ID
# ============================================================

def make_chunk_id(
    pdf_name,
    page,
    idx,
    text_value
):

    digest = hashlib.sha1(
        (
            f"{pdf_name}-"
            f"{page}-"
            f"{idx}-"
            f"{text_value[:80]}"
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    return (
        f"{pdf_name}_"
        f"p{page}_"
        f"c{idx}_"
        f"{digest}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PROMOTION PDF INGESTION"
    )

    print(
        "Improved table-aware OCR"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    client = configure_gemini()

    # --------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------

    if not check_tesseract():

        sys.exit(1)

    # --------------------------------------------------------
    # Find PDFs
    # --------------------------------------------------------

    pdf_paths = sorted(
        glob.glob(
            os.path.join(
                PDF_DIR,
                "*.pdf"
            )
        )
    )

    if not pdf_paths:

        print(
            "\nNo PDFs found in:"
        )

        print(
            PDF_DIR
        )

        sys.exit(1)

    print(
        f"\nFound {len(pdf_paths)} PDF(s):"
    )

    for pdf in pdf_paths:

        print(
            f"  - "
            f"{os.path.basename(pdf)}"
        )

    # --------------------------------------------------------
    # CSV directory
    # --------------------------------------------------------

    os.makedirs(
        CSV_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Rebuild Chroma
    # --------------------------------------------------------

    chroma_client = (
        chromadb.PersistentClient(
            path=CHROMA_DIR
        )
    )

    try:

        chroma_client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "\nOld search collection removed."
        )

    except Exception:

        pass

    collection = (
        chroma_client.create_collection(

            name=COLLECTION_NAME,

            metadata={
                "hnsw:space":
                    "cosine"
            }
        )
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_ids = []

    all_documents = []

    all_metadata = []

    all_pickle_records = []

    total_employee_rows = 0


    # ========================================================
    # PROCESS EVERY PDF
    # ========================================================

    for pdf_path in pdf_paths:

        pdf_name = os.path.basename(
            pdf_path
        )

        print(
            "\n"
            +
            "-" * 70
        )

        print(
            f"Processing: {pdf_name}"
        )

        print(
            "-" * 70
        )

        doc = pymupdf.open(
            pdf_path
        )

        current_goo_reference = (
            "unknown reference"
        )

        # ----------------------------------------------------
        # EVERY PAGE
        # ----------------------------------------------------

        for page_index, page in enumerate(
            doc
        ):

            page_number = (
                page_index + 1
            )

            print(
                f"\nPage "
                f"{page_number}/"
                f"{len(doc)}"
            )

            # ------------------------------------------------
            # Render page
            # ------------------------------------------------

            rgb_image = render_page_image(
                page
            )

            # ------------------------------------------------
            # Clean image for OCR
            # ------------------------------------------------

            cleaned_image = (
                remove_gridlines(
                    rgb_image
                )
            )

            # ------------------------------------------------
            # OCR words
            # ------------------------------------------------

            words = ocr_words(
                cleaned_image
            )

            if not words:

                print(
                    "  No OCR words found."
                )

                continue

            # ------------------------------------------------
            # Group OCR words
            # ------------------------------------------------

            rows = group_into_rows(
                words
            )

            # ------------------------------------------------
            # Reconstruct table
            # ------------------------------------------------

            table_rows = reconstruct_table(
                rows,
                rgb_image
            )

            # ------------------------------------------------
            # G.O. reference
            # ------------------------------------------------

            plain_preview = " ".join(
                word["text"]
                for word in words[:250]
            )

            detected_reference = (
                find_goo_reference(
                    plain_preview
                )
            )

            if (
                detected_reference
                !=
                "unknown reference"
            ):

                current_goo_reference = (
                    detected_reference
                )

            # =================================================
            # TABLE PAGE
            # =================================================

            if table_rows:

                print(
                    f"  Employee rows found: "
                    f"{len(table_rows)}"
                )

                # ------------------------------------------------
                # CSV
                # ------------------------------------------------

                csv_path = os.path.join(
                    CSV_DIR,
                    (
                        f"{os.path.splitext(pdf_name)[0]}"
                        f"_page{page_number}.csv"
                    )
                )

                with open(
                    csv_path,
                    "w",
                    newline="",
                    encoding="utf-8-sig"
                ) as file:

                    writer = csv.DictWriter(
                        file,
                        fieldnames=COLUMN_LABELS
                    )

                    writer.writeheader()

                    for record in table_rows:

                        writer.writerow(
                            record
                        )

                # ------------------------------------------------
                # Employee chunks
                # ------------------------------------------------

                for row_index, record in enumerate(
                    table_rows
                ):

                    name_value = clean_value(
                        record.get(
                            NAME_COLUMN,
                            ""
                        )
                    )

                    employee_id = clean_value(
                        record.get(
                            ID_COLUMN,
                            ""
                        )
                    )

                    place_value = clean_value(
                        record.get(
                            PLACE_COLUMN,
                            ""
                        )
                    )

                    reference_value = clean_value(
                        record.get(
                            REFERENCE_COLUMN,
                            ""
                        )
                    )

                    revised_date_value = clean_value(
                        record.get(
                            REVISED_DATE_COLUMN,
                            ""
                        )
                    )

                    print(
                        "\n  Employee:"
                    )

                    print(
                        f"    Name       : "
                        f"{name_value}"
                    )

                    print(
                        f"    Place      : "
                        f"{place_value}"
                    )

                    print(
                        f"    Employee ID: "
                        f"{employee_id}"
                    )

                    print(
                        f"    Reference  : "
                        f"{reference_value}"
                    )

                    print(
                        f"    Revised    : "
                        f"{revised_date_value}"
                    )

                    chunk_text_value = (
                        row_to_chunk_text(
                            pdf_name,
                            page_number,
                            current_goo_reference,
                            record
                        )
                    )

                    chunk_id = make_chunk_id(
                        pdf_name,
                        page_number,
                        row_index,
                        chunk_text_value
                    )

                    all_ids.append(
                        chunk_id
                    )

                    all_documents.append(
                        chunk_text_value
                    )

                    # --------------------------------------------
                    # Store all fields in Chroma
                    # --------------------------------------------

                    metadata = {

                        "source":
                            pdf_name,

                        "page":
                            page_number,

                        "row_type":
                            "table_row",

                        "s_no":
                            clean_value(
                                record.get(
                                    "S.No",
                                    ""
                                )
                            ),

                        "name":
                            name_value,

                        "name_lower":
                            name_value.lower(),

                        "employee_name":
                            name_value,

                        "emp_id":
                            employee_id,

                        "employee_id":
                            employee_id,

                        "present_place":
                            place_value,

                        "present_place_of_working":
                            place_value,

                        "reference":
                            reference_value,

                        "promotion_reference":
                            reference_value,

                        "reference_of_promotion_order":
                            reference_value,

                        "revised_date":
                            revised_date_value,

                        "now_revised_date":
                            revised_date_value,

                        "now_revised_date_of_promotion":
                            revised_date_value,

                        "goo_reference":
                            current_goo_reference
                    }

                    all_metadata.append(
                        metadata
                    )

                    # --------------------------------------------
                    # Pickle
                    # --------------------------------------------

                    pickle_record = (
                        create_pickle_record(
                            pdf_name,
                            page_number,
                            record,
                            chunk_text_value,
                            current_goo_reference
                        )
                    )

                    all_pickle_records.append(
                        pickle_record
                    )

                    total_employee_rows += 1

            # =================================================
            # NARRATIVE PAGE
            # =================================================

            else:

                narrative_text = (
                    ocr_plain_text(
                        cleaned_image
                    )
                )

                if not narrative_text:

                    continue

                print(
                    f"  Narrative page: "
                    f"{len(narrative_text)} "
                    f"characters"
                )

                narrative_chunks = chunk_text(
                    narrative_text
                )

                for chunk_index, chunk in enumerate(
                    narrative_chunks
                ):

                    chunk_id = make_chunk_id(
                        pdf_name,
                        page_number,
                        chunk_index,
                        chunk
                    )

                    all_ids.append(
                        chunk_id
                    )

                    all_documents.append(
                        chunk
                    )

                    all_metadata.append(
                        {
                            "source":
                                pdf_name,

                            "page":
                                page_number,

                            "row_type":
                                "narrative",

                            "s_no":
                                "",

                            "name":
                                "",

                            "name_lower":
                                "",

                            "employee_name":
                                "",

                            "emp_id":
                                "",

                            "employee_id":
                                "",

                            "present_place":
                                "",

                            "present_place_of_working":
                                "",

                            "reference":
                                "",

                            "promotion_reference":
                                "",

                            "reference_of_promotion_order":
                                "",

                            "revised_date":
                                "",

                            "now_revised_date":
                                "",

                            "now_revised_date_of_promotion":
                                "",

                            "goo_reference":
                                current_goo_reference
                        }
                    )

                    all_pickle_records.append(
                        {
                            "text":
                                chunk,

                            "source":
                                pdf_name,

                            "page":
                                page_number,

                            "row_type":
                                "narrative",

                            "s_no":
                                "",

                            "name":
                                "",

                            "name_lower":
                                "",

                            "employee_name":
                                "",

                            "emp_id":
                                "",

                            "employee_id":
                                "",

                            "present_place":
                                "",

                            "present_place_of_working":
                                "",

                            "reference":
                                "",

                            "promotion_reference":
                                "",

                            "reference_of_promotion_order":
                                "",

                            "revised_date":
                                "",

                            "now_revised_date":
                                "",

                            "now_revised_date_of_promotion":
                                "",

                            "goo_reference":
                                current_goo_reference
                        }
                    )

        doc.close()


    # ========================================================
    # CHECK
    # ========================================================

    if not all_documents:

        print(
            "\nERROR: Nothing was extracted."
        )

        sys.exit(1)


    # ========================================================
    # SAVE PICKLE
    # ========================================================

    print(
        "\nSaving chunks.pkl..."
    )

    with open(
        PICKLE_FILE,
        "wb"
    ) as file:

        pickle.dump(
            all_pickle_records,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"Saved "
        f"{len(all_pickle_records)} "
        f"records."
    )


    # ========================================================
    # EXTRACTION SUMMARY
    # ========================================================

    employee_records = [

        record

        for record
        in all_pickle_records

        if record.get(
            "row_type"
        )
        ==
        "table_row"
    ]

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        f"Employee rows extracted: "
        f"{len(employee_records)}"
    )

    print(
        f"Total chunks: "
        f"{len(all_documents)}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # SAMPLE RECORDS
    # ========================================================

    print(
        "\nSample extracted records:"
    )

    for record in employee_records[:10]:

        print(
            "\n"
            f"Name       : "
            f"{record.get('name', '')}"
        )

        print(
            f"Place      : "
            f"{record.get('present_place', '')}"
        )

        print(
            f"Employee ID: "
            f"{record.get('emp_id', '')}"
        )

        print(
            f"Reference  : "
            f"{record.get('reference_of_promotion_order', '')}"
        )

        print(
            f"Revised    : "
            f"{record.get('now_revised_date_of_promotion', '')}"
        )

        print(
            f"PDF        : "
            f"{record.get('source', '')}"
        )

        print(
            f"Page       : "
            f"{record.get('page', '')}"
        )

        print(
            "-" * 60
        )


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    total_chunks = len(
        all_documents
    )

    print(
        f"\nCreating embeddings for "
        f"{total_chunks} chunks..."
    )

    for start in range(
        0,
        total_chunks,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_chunks
        )

        print(
            f"\nEmbedding "
            f"{start + 1}-"
            f"{end} of "
            f"{total_chunks}"
        )

        batch_embeddings = embed_texts(
            client,
            all_documents[
                start:end
            ]
        )

        collection.add(

            ids=all_ids[
                start:end
            ],

            documents=all_documents[
                start:end
            ],

            metadatas=all_metadata[
                start:end
            ],

            embeddings=batch_embeddings
        )

        print(
            f"Stored "
            f"{start + 1}-"
            f"{end}"
        )

        if end < total_chunks:

            time.sleep(
                BATCH_DELAY
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"PDFs          : "
        f"{len(pdf_paths)}"
    )

    print(
        f"Employee rows : "
        f"{total_employee_rows}"
    )

    print(
        f"Total chunks  : "
        f"{len(all_documents)}"
    )

    print(
        f"Search records: "
        f"{collection.count()}"
    )

    print(
        f"CSV directory : "
        f"{CSV_DIR}"
    )

    print(
        f"Pickle file   : "
        f"{PICKLE_FILE}"
    )

    print(
        "=" * 70
    )

    print(
        "\nTesseract:"
    )

    print(
        TESSERACT_PATH
    )

    print(
        "\nNow run:"
    )

    print(
        r"C:\Users\hp\miniconda3\python.exe -m streamlit run C:\Users\hp\PycharmProjects\pr-project\app.py"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()