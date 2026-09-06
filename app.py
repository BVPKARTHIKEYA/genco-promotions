import os
import re
import pickle

import streamlit as st
from dotenv import load_dotenv
import chromadb
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
PICKLE_FILE = os.path.join(BASE_DIR, "chunks.pkl")
PDF_DIR = os.path.join(BASE_DIR, "knowledge-base")


# ============================================================
# MODELS
# ============================================================

COLLECTION_NAME = "knowledge_base"

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"


# ============================================================
# SEARCH SETTINGS
# ============================================================

TOP_K_SEMANTIC = 8
DISTANCE_THRESHOLD = 0.7

NO_DETAILS_MESSAGE = "No details are found."


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Promotion Details",
    page_icon="📋",
    layout="wide",
)


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY is not configured.")
        st.stop()

    try:
        return genai.Client(api_key=api_key)
    except Exception:
        st.error("Unable to connect to the search service.")
        st.stop()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource(show_spinner=False)
def get_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


# ============================================================
# LOAD PICKLE
# ============================================================

@st.cache_data(show_spinner=False)
def load_pickle_chunks():
    if not os.path.exists(PICKLE_FILE):
        return []

    try:
        with open(PICKLE_FILE, "rb") as file:
            data = pickle.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, tuple):
            return list(data)

        return []
    except Exception:
        return []


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "be",
    "by",
    "can",
    "could",
    "details",
    "detail",
    "did",
    "do",
    "does",
    "employee",
    "employees",
    "find",
    "for",
    "from",
    "give",
    "get",
    "has",
    "have",
    "how",
    "in",
    "information",
    "is",
    "individual",
    "individuals",
    "me",
    "name",
    "names",
    "of",
    "on",
    "people",
    "person",
    "persons",
    "please",
    "promotion",
    "promotions",
    "promoted",
    "show",
    "tell",
    "the",
    "their",
    "them",
    "to",
    "what",
    "were",
    "who",
    "with",
    "whose",
    "you",
    "record",
    "records",
    "this",
    "that",
    "place",
    "working",
    "reference",
    "order",
    "date",
    "revised",
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# CLEAN EXTRACTED VALUE
# ============================================================

def clean_extracted_value(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()

    # Remove surrounding quotation marks repeatedly.
    while len(value) >= 2 and (
        value[0] in "'\"“‘`"
        and value[-1] in "'\"”’`"
    ):
        value = value[1:-1].strip()

    # Remove remaining quote characters at the ends.
    value = re.sub(r"^[\"'“‘`]+", "", value)
    value = re.sub(r"[\"'”’`]+$", "", value)

    # Remove OCR/table separators at the ends.
    value = value.strip(" :;-|\t*_")
    value = value.strip("\"'“”‘’`")

    return value.strip()


# ============================================================
# CLEAN PROMOTION REFERENCE
# ============================================================

def clean_promotion_reference(value):
    value = clean_extracted_value(value)

    # Remove accidental punctuation left by OCR.
    value = value.rstrip(" ,;:-|.")

    return value.strip()


# ============================================================
# CLEAN REVISED DATE
# ============================================================

def clean_revised_date(value):
    value = clean_extracted_value(value)

    # Remove OCR punctuation, but do not remove date separators
    # inside the value.
    value = value.rstrip(" ,;:-|.")

    return value.strip()


# ============================================================
# SEARCH TERMS
# ============================================================

def extract_candidate_terms(query):
    query = str(query).strip()

    if not query:
        return []

    cleaned = re.sub(r"[^\w\s.]", " ", query.lower())

    words = [
        word.strip(".")
        for word in cleaned.split()
        if word.strip(".")
    ]

    words = [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    terms = []

    # Individual useful terms.
    for word in words:
        if word not in terms:
            terms.append(word)

    # Two-word phrases are useful for employee names such as
    # "Ravi Kumar".
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if phrase not in terms:
            terms.append(phrase)

    return terms


# ============================================================
# RECURSIVE VALUE SEARCH
# ============================================================

def find_value_recursive(item, possible_keys):
    if isinstance(item, dict):
        # Exact key match first.
        for key in possible_keys:
            if key in item:
                value = clean_extracted_value(item[key])
                if value:
                    return value

        # Case-insensitive key match.
        for existing_key, value in item.items():
            existing_normalized = normalize_text(existing_key)

            for wanted_key in possible_keys:
                wanted_normalized = normalize_text(wanted_key)

                if existing_normalized == wanted_normalized:
                    value = clean_extracted_value(value)

                    if value:
                        return value

        # Search nested values.
        for value in item.values():
            if isinstance(value, (dict, list, tuple)):
                result = find_value_recursive(value, possible_keys)

                if result:
                    return result

    elif isinstance(item, (list, tuple)):
        for value in item:
            result = find_value_recursive(value, possible_keys)

            if result:
                return result

    return ""


# ============================================================
# VALIDATE EMPLOYEE NAME
# ============================================================

def is_valid_employee_name(name):
    if not name:
        return False

    name = clean_extracted_value(name)

    if not name:
        return False

    normalized = normalize_text(name)

    invalid_phrases = [
        "through",
        "copy to",
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
        "promotion orders",
        "revised date",
        "now revised",
        "present place",
        "place of working",
        "employee id",
        "emp id",
        "the chief",
        "chief",
        "subject",
        "signature",
        "copy furnished",
        "copies to",
    ]

    for phrase in invalid_phrases:
        if phrase in normalized:
            return False

    words = name.split()

    if len(words) > 7:
        return False

    if len(name) > 80:
        return False

    punctuation_count = len(re.findall(r"[,;:/|]", name))

    if punctuation_count >= 3:
        return False

    if not re.search(r"[A-Za-z]", name):
        return False

    return True


# ============================================================
# EXTRACT EMPLOYEE NAME FROM TEXT
# ============================================================

def extract_employee_name(text):
    if not text:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        (
            r"Name\s+of\s+the\s+employee"
            r"\s*[:\-]?\s*(.*?)"
            r"(?=\s+Present\s+place"
            r"|\s+Emp\s*ID"
            r"|\s+Reference"
            r"|\s+Now\s+Revised"
            r"|$)"
        ),
        (
            r"Name\s+of\s+employee"
            r"\s*[:\-]?\s*(.*?)"
            r"(?=\s+Present\s+place"
            r"|\s+Emp\s*ID"
            r"|\s+Reference"
            r"|\s+Now\s+Revised"
            r"|$)"
        ),
        (
            r"Employee\s+Name"
            r"\s*[:\-]?\s*(.*?)"
            r"(?=\s+Present\s+place"
            r"|\s+Emp\s*ID"
            r"|\s+Reference"
            r"|\s+Now\s+Revised"
            r"|$)"
        ),
        (
            r"\bName\s*[:\-]\s*(.*?)"
            r"(?=\s+Present\s+place"
            r"|\s+Emp\s*ID"
            r"|\s+Reference"
            r"|\s+Now\s+Revised"
            r"|$)"
        ),
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            name = clean_extracted_value(match.group(1))

            if is_valid_employee_name(name):
                return name

    # Pipe-separated row.
    if "|" in text:
        parts = [
            clean_extracted_value(x)
            for x in text.split("|")
        ]

        if len(parts) >= 3:
            candidate = parts[1]

            if is_valid_employee_name(candidate):
                return candidate

    # Tab-separated row.
    if "\t" in text:
        parts = [
            clean_extracted_value(x)
            for x in text.split("\t")
        ]

        if len(parts) >= 3:
            candidate = parts[1]

            if is_valid_employee_name(candidate):
                return candidate

    return ""


# ============================================================
# GET EMPLOYEE NAME
# ============================================================

def get_employee_name(item):
    keys = [
        "name",
        "employee_name",
        "employeeName",
        "Employee Name",
        "Employee_Name",
        "Name of the employee",
        "Name of employee",
        "Name",
        "NAME",
    ]

    name = find_value_recursive(item, keys)

    if is_valid_employee_name(name):
        return clean_extracted_value(name)

    text = get_text(item)

    name = extract_employee_name(text)

    if is_valid_employee_name(name):
        return name

    return ""


# ============================================================
# GET EMPLOYEE ID
# ============================================================

def get_employee_id(item):
    keys = [
        "emp_id",
        "employee_id",
        "employeeId",
        "Employee ID",
        "Employee_ID",
        "Emp ID.No",
        "Emp ID No",
        "Emp ID",
        "EmpID",
        "EMP ID",
    ]

    value = find_value_recursive(item, keys)

    if value:
        return clean_extracted_value(value)

    text = get_text(item)

    if not text:
        return ""

    patterns = [
        (
            r"Emp\s*ID\.?\s*No\.?"
            r"\s*[:\-]?\s*"
            r"([A-Za-z0-9\/\-_]+)"
        ),
        (
            r"Emp\s*ID\s*No\.?"
            r"\s*[:\-]?\s*"
            r"([A-Za-z0-9\/\-_]+)"
        ),
        (
            r"Employee\s*ID"
            r"\s*[:\-]?\s*"
            r"([A-Za-z0-9\/\-_]+)"
        ),
        (
            r"Emp\s*ID"
            r"\s*[:\-]?\s*"
            r"([A-Za-z0-9\/\-_]+)"
        ),
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return clean_extracted_value(match.group(1))

    return ""


# ============================================================
# GET PRESENT PLACE
# ============================================================

def get_present_place(item):
    keys = [
        "present_place",
        "present_place_of_working",
        "present_place_of_work",
        "Present place of working",
        "Present Place of Working",
        "Present place",
        "place_of_working",
        "working_place",
    ]

    value = find_value_recursive(item, keys)

    if value:
        return clean_extracted_value(value)

    text = get_text(item)

    if not text:
        return ""

    pattern = (
        r"Present\s+place\s+of\s+working"
        r"\s*[:\-]?\s*(.*?)"
        r"(?=\s+Emp\s*ID"
        r"|\s+Reference"
        r"|\s+Now\s+Revised"
        r"|$)"
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return clean_extracted_value(match.group(1))

    if "|" in text:
        parts = [
            clean_extracted_value(x)
            for x in text.split("|")
        ]

        if len(parts) >= 3:
            return parts[2]

    return ""


# ============================================================
# GET PROMOTION REFERENCE
# ============================================================

def get_promotion_reference(item):
    keys = [
        "promotion_reference",
        "reference_of_promotion_order",
        "reference_of_promotion_orders",
        "promotion_order_reference",
        "reference",
        "Reference of Promotion orders",
        "Reference of Promotion Order",
        "goo_reference",
    ]

    value = find_value_recursive(item, keys)

    if value:
        return clean_promotion_reference(value)

    text = get_text(item)

    if not text:
        return ""

    pattern = (
        r"Reference\s+of\s+Promotion"
        r"(?:\s+orders?)?"
        r"\s*[:\-]?\s*(.*?)"
        r"(?=\s+Now\s+Revised"
        r"|$)"
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return clean_promotion_reference(match.group(1))

    if "|" in text:
        parts = [
            clean_promotion_reference(x)
            for x in text.split("|")
        ]

        if len(parts) >= 5:
            return parts[4]

    return ""


# ============================================================
# GET REVISED DATE
# ============================================================

def get_revised_date(item):
    keys = [
        "revised_date",
        "now_revised_date",
        "now_revised_date_of_promotion",
        "promotion_revised_date",
        "Now Revised date of promotion",
        "Now Revised Date",
        "revised_promotion_date",
        "date_of_promotion",
    ]

    value = find_value_recursive(item, keys)

    if value:
        return clean_revised_date(value)

    text = get_text(item)

    if not text:
        return ""

    pattern = (
        r"Now\s+Revised"
        r"\s+date\s+of\s+promotion"
        r"(?:\s+as\s+.*?terms\s+of\s+Merit"
        r"\s+based\s+Seniority"
        r"(?:\s+with\s+effect\s+from)?)?"
        r"\s*[:\-]?\s*(.*)"
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return clean_revised_date(match.group(1))

    if "|" in text:
        parts = [
            clean_revised_date(x)
            for x in text.split("|")
        ]

        if len(parts) >= 6:
            return parts[5]

    return ""


# ============================================================
# GET SOURCE
# ============================================================

def get_source(item):
    keys = [
        "source",
        "pdf",
        "pdf_name",
        "file",
        "filename",
        "source_pdf",
    ]

    value = find_value_recursive(item, keys)

    return value if value else "Unknown"


# ============================================================
# GET PAGE
# ============================================================

def get_page(item):
    keys = [
        "page",
        "page_number",
        "page_no",
        "page_num",
    ]

    value = find_value_recursive(item, keys)

    return value if value else "?"


# ============================================================
# GET TEXT
# ============================================================

def get_text(item):
    if not isinstance(item, dict):
        return ""

    for key in [
        "text",
        "content",
        "document",
        "chunk",
        "raw_text",
        "ocr_text",
    ]:
        value = item.get(key, "")

        if value:
            return str(value)

    return ""


# ============================================================
# GET ROW TYPE
# ============================================================

def get_row_type(item):
    if not isinstance(item, dict):
        return "table_row"

    return str(
        item.get("row_type", "table_row")
    ).lower()


# ============================================================
# DIRECT NAME SEARCH
# ============================================================

def search_by_name(chunks, query):
    if not chunks:
        return []

    terms = extract_candidate_terms(query)

    if not terms:
        return []

    matches = []
    seen = set()

    for item in chunks:
        if not isinstance(item, (dict, list, tuple)):
            continue

        name = get_employee_name(item)

        # Ignore invalid OCR rows.
        if not is_valid_employee_name(name):
            continue

        name_normalized = normalize_text(name)

        matched = False

        # IMPORTANT:
        # Search only inside the employee name.
        # This prevents matches such as "jaya" in "Vijayawada".
        for term in terms:
            term_normalized = normalize_text(term)

            if (
                term_normalized
                and term_normalized in name_normalized
            ):
                matched = True
                break

        if not matched:
            continue

        emp_id = get_employee_id(item)
        present_place = get_present_place(item)
        promotion_reference = get_promotion_reference(item)
        revised_date = get_revised_date(item)
        source = get_source(item)
        page = get_page(item)
        text = get_text(item)

        key = (
            normalize_text(name),
            normalize_text(emp_id),
            normalize_text(source),
            str(page),
        )

        if key in seen:
            continue

        seen.add(key)

        matches.append(
            {
                "name": name,
                "present_place": present_place,
                "emp_id": emp_id,
                "promotion_reference": promotion_reference,
                "revised_date": revised_date,
                "source": source,
                "page": page,
                "text": text,
                "row_type": get_row_type(item),
                "keyword_match": True,
                "distance": 0.0,
            }
        )

    return matches


# ============================================================
# GEMINI EMBEDDING
# ============================================================

def embed_query(client, query):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config={
            "task_type": "RETRIEVAL_QUERY",
        },
    )

    return result.embeddings[0].values


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(client, collection, query, top_k=TOP_K_SEMANTIC):
    if collection is None:
        return []

    try:
        query_embedding = embed_query(client, query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        if not docs:
            return []

        docs = docs[0]
        metas = metas[0] if metas else []
        distances = distances[0] if distances else []

        hits = []

        for index, doc in enumerate(docs):
            meta = (
                metas[index]
                if index < len(metas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else 999
            )

            if distance > DISTANCE_THRESHOLD:
                continue

            meta = meta or {}

            temp_item = {
                "text": doc,
                "name": meta.get("name", ""),
                "employee_name": meta.get("employee_name", ""),
                "emp_id": meta.get("emp_id", ""),
                "employee_id": meta.get("employee_id", ""),
                "present_place": meta.get("present_place", ""),
                "present_place_of_working": meta.get(
                    "present_place_of_working", ""
                ),
                "promotion_reference": meta.get(
                    "promotion_reference", ""
                ),
                "reference_of_promotion_order": meta.get(
                    "reference_of_promotion_order", ""
                ),
                "revised_date": meta.get("revised_date", ""),
                "now_revised_date": meta.get(
                    "now_revised_date", ""
                ),
                "now_revised_date_of_promotion": meta.get(
                    "now_revised_date_of_promotion", ""
                ),
            }

            name = get_employee_name(temp_item)
            emp_id = get_employee_id(temp_item)
            present_place = get_present_place(temp_item)
            promotion_reference = get_promotion_reference(
                temp_item
            )
            revised_date = get_revised_date(temp_item)

            # Do not display invalid/non-employee OCR rows.
            if not name or not is_valid_employee_name(name):
                continue

            hits.append(
                {
                    "name": name,
                    "present_place": present_place,
                    "emp_id": emp_id,
                    "promotion_reference": promotion_reference,
                    "revised_date": revised_date,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "?"),
                    "text": doc,
                    "keyword_match": False,
                    "distance": distance,
                }
            )

        return hits

    except Exception:
        return []


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(query, context_rows):
    context_parts = []

    for index, row in enumerate(context_rows, start=1):
        context_parts.append(
            f"""
[CONTEXT {index}]

Employee Name:
{row.get("name", "")}

Present Place of Working:
{row.get("present_place", "")}

Employee ID:
{row.get("emp_id", "")}

Reference of Promotion Order:
{row.get("promotion_reference", "")}

Revised Date of Promotion:
{row.get("revised_date", "")}

Source PDF:
{row.get("source", "Unknown")}

Page:
{row.get("page", "?")}

Text:
{row.get("text", "")}
"""
        )

    context_block = "\n".join(context_parts)

    return f"""
You are a promotion records assistant.

Answer ONLY using the supplied promotion-record context.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent information.
4. Preserve names exactly as provided.
5. Preserve employee IDs exactly as provided.
6. Preserve dates exactly as provided.
7. Do not mix different employees.
8. If multiple employees are supported, include all of them.
9. If information is missing, do not invent it.
10. Do not mention internal databases.
11. Do not mention ChromaDB.
12. Do not mention embeddings.
13. Do not mention implementation details.
14. If there is not enough information, return exactly:
No details are found.

CONTEXT:

{context_block}

USER QUESTION:

{query}

ANSWER:
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(client, prompt):
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
        )

        answer = (
            response.text
            or ""
        ).strip()

        if not answer:
            return NO_DETAILS_MESSAGE

        return answer

    except Exception:
        return NO_DETAILS_MESSAGE


# ============================================================
# PDF PATH
# ============================================================

def get_pdf_path(source):
    if not source:
        return None

    filename = os.path.basename(str(source))

    path = os.path.join(PDF_DIR, filename)

    if os.path.isfile(path):
        return path

    return None


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

def create_results_table(hits):
    table_rows = []
    seen = set()

    for hit in hits:
        name = clean_extracted_value(
            get_employee_name(hit)
        )

        # Never show invalid names.
        if not is_valid_employee_name(name):
            continue

        present_place = clean_extracted_value(
            get_present_place(hit)
        )

        emp_id = clean_extracted_value(
            get_employee_id(hit)
        )

        promotion_reference = clean_promotion_reference(
            get_promotion_reference(hit)
        )

        revised_date = clean_revised_date(
            get_revised_date(hit)
        )

        source = get_source(hit)
        page = get_page(hit)

        key = (
            normalize_text(name),
            normalize_text(emp_id),
            normalize_text(source),
            str(page),
        )

        if key in seen:
            continue

        seen.add(key)

        table_rows.append(
            {
                "Employee Name": name,
                "Present Place of Working": (
                    present_place
                    if present_place
                    else "Not available"
                ),
                "Employee ID": (
                    emp_id
                    if emp_id
                    else "Not available"
                ),
                "Reference of Promotion Order": (
                    promotion_reference
                    if promotion_reference
                    else "Not available"
                ),
                "Revised Date of Promotion": (
                    revised_date
                    if revised_date
                    else "Not available"
                ),
                "PDF": source,
                "Page": page,
            }
        )

    return table_rows


# ============================================================
# DISPLAY TABLE
# ============================================================

def display_results_table(hits):
    table_data = create_results_table(hits)

    if not table_data:
        st.write(NO_DETAILS_MESSAGE)
        return

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "Employee Name": st.column_config.TextColumn(
                "Employee Name",
                width="medium",
            ),
            "Present Place of Working": st.column_config.TextColumn(
                "Present Place of Working",
                width="large",
            ),
            "Employee ID": st.column_config.TextColumn(
                "Employee ID",
                width="medium",
            ),
            "Reference of Promotion Order": st.column_config.TextColumn(
                "Reference of Promotion Order",
                width="large",
            ),
            "Revised Date of Promotion": st.column_config.TextColumn(
                "Revised Date of Promotion",
                width="large",
            ),
            "PDF": st.column_config.TextColumn(
                "PDF",
                width="small",
            ),
            "Page": st.column_config.TextColumn(
                "Page",
                width="small",
            ),
        },
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_relevant_sources(hits):
    if not hits:
        return

    st.subheader("Relevant Sources")

    seen = set()
    source_index = 0

    for hit in hits:
        source = get_source(hit)
        page = get_page(hit)

        key = (
            normalize_text(source),
            str(page),
        )

        if key in seen:
            continue

        seen.add(key)
        source_index += 1

        st.markdown(
            f"**{source_index}. {source} — Page {page}**"
        )

        pdf_path = get_pdf_path(source)

        if pdf_path:
            try:
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label=f"📄 Open / Download {source}",
                    data=pdf_bytes,
                    file_name=os.path.basename(source),
                    mime="application/pdf",
                    key=f"pdf_{source_index}_{source}_{page}",
                )

            except Exception:
                st.caption("PDF could not be opened.")
        else:
            st.caption("PDF file is not available locally.")

        st.divider()


# ============================================================
# MAIN
# ============================================================

def main():
    st.title("📋 Promotion Details")

    st.caption(
        "Search promotion records by employee name or ask a question."
    )

    chunks = load_pickle_chunks()
    collection = get_collection()

    if not chunks and collection is None:
        st.error("No searchable data is available.")
        st.stop()

    # --------------------------------------------------------
    # SEARCH BOX
    # --------------------------------------------------------

    query = st.text_input(
        "Search",
        placeholder="Example: Jaya",
    )

    search_clicked = st.button(
        "🔍 Search",
        type="primary",
    )

    if not search_clicked:
        return

    query = query.strip()

    if not query:
        st.info("Please enter a name or question.")
        return

    # --------------------------------------------------------
    # DIRECT NAME SEARCH
    # --------------------------------------------------------

    with st.spinner("Searching..."):
        name_hits = search_by_name(
            chunks,
            query,
        )

    # --------------------------------------------------------
    # NAME RESULTS
    # --------------------------------------------------------

    if name_hits:
        st.subheader("Results")
        display_results_table(name_hits)
        display_relevant_sources(name_hits)
        return

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    if collection is None:
        st.subheader("Results")
        st.write(NO_DETAILS_MESSAGE)
        return

    with st.spinner("Searching relevant records..."):
        context_rows = semantic_search(
            get_gemini_client(),
            collection,
            query,
        )

    if not context_rows:
        st.subheader("Results")
        st.write(NO_DETAILS_MESSAGE)
        return

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner("Preparing results..."):
        client = get_gemini_client()

        prompt = build_prompt(
            query,
            context_rows,
        )

        answer = generate_answer(
            client,
            prompt,
        )

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("Results")

    if (
        not answer
        or answer.lower().strip()
        == NO_DETAILS_MESSAGE.lower()
    ):
        st.write(NO_DETAILS_MESSAGE)
    else:
        st.markdown(answer)

    display_relevant_sources(context_rows)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
