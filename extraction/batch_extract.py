"""
NERC Batch Extraction — Scrape all quarterly PDFs into a single raw JSON
=========================================================================
Scans all NERC quarterly report PDFs, extracts target tables using
content-based page finding + Claude API, outputs nerc_raw_extract.json.

Handles format variation across 2017-2026 vintages.
Skips annual reports and duplicate files.
Supports resume: re-run safely, already-extracted reports are skipped.

Usage:
    ANTHROPIC_API_KEY=sk-... python batch_extract.py
    ANTHROPIC_API_KEY=sk-... python batch_extract.py --dry-run   # list files, no extraction

Output: nerc_raw_extract.json
Requires: pip install pdfplumber anthropic
"""

import pdfplumber
import anthropic
import re
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

REPORTS_DIR = "Disequilibrium/NERC REPORTS"
OUTPUT_FILE = "nerc_raw_extract.json"
MODEL = "claude-sonnet-4-6"  # or "claude-haiku-4-5-20251001" for lower cost

# ── Period detection ───────────────────────────────────────────────────────────

QUARTER_WORDS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4}

def detect_period_from_filename(filename):
    """
    Extract year and quarter from filename. Returns (year, quarter) or (None, None).
    Handles:
      2018Q1_..., 2025_Q4-..., nerc-quarterly-report-q3-2017,
      NERC Second Quarter Report 2018, NERCThirdQuarter2023Report, etc.
    """
    name = filename.upper().replace("-", " ").replace("_", " ")

    # Pattern: 2018Q1, 2025 Q4, 2019Q2
    m = re.search(r'(\d{4})\s*Q(\d)', name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern: Q3 2017
    m = re.search(r'Q(\d)\s*(\d{4})', name)
    if m:
        return int(m.group(2)), int(m.group(1))

    # Pattern: FIRST QUARTER 2018
    m = re.search(r'(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\s+(\d{4})', name)
    if m:
        return int(m.group(2)), QUARTER_WORDS[m.group(1)]

    # Pattern: "NERC Second Quarter Report 2018", "NERCSecondQuarterReport2023"
    # Handle with/without spaces between words
    for word, qnum in QUARTER_WORDS.items():
        if word in name:
            year_match = re.search(r'(20\d{2})', name)
            if year_match:
                return int(year_match.group(1)), qnum

    return None, None


def detect_period_from_pdf(pdf):
    """
    Scan first pages of PDF for quarter identification.
    Uses title page and headers, not comparison tables that mention prior quarters.
    """
    for i in range(min(5, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ""
        # Look for the report's own quarter declaration (usually on cover/title)
        m = re.search(r"(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\s+(20\d{2})", text, re.IGNORECASE)
        if m:
            return int(m.group(2)), QUARTER_WORDS[m.group(1).upper()]
    return None, None


def detect_period(filename, pdf):
    """Detect period: prefer filename, fall back to PDF content."""
    year, quarter = detect_period_from_filename(filename)
    if year and quarter:
        return year, quarter
    return detect_period_from_pdf(pdf)


# ── File filtering ─────────────────────────────────────────────────────────────

def is_annual_report(filename):
    """Skip annual reports — they have different table formats."""
    name = filename.upper()
    if "ANNUAL" in name:
        return True
    return False


def is_duplicate(filename, all_files):
    """
    Skip files with (1) suffix if a non-(1) version exists.
    Exception: 2026_Q1-Report (1).pdf has no clean version.
    """
    if "(1)" not in filename:
        return False
    clean_name = filename.replace(" (1)", "").replace("(1)", "")
    if clean_name in all_files:
        return True
    return False


def get_quarterly_pdfs(reports_dir):
    """Return sorted list of quarterly report PDFs to process."""
    all_files = os.listdir(reports_dir)
    pdfs = [f for f in all_files if f.lower().endswith(".pdf")]
    result = []
    for f in sorted(pdfs):
        if is_annual_report(f):
            continue
        if is_duplicate(f, all_files):
            continue
        result.append(f)
    return result


# ── Page scanning utilities ────────────────────────────────────────────────────

def is_toc_page(text):
    dot_leaders = text.count("...") + text.count(". . .")
    if dot_leaders > 5:
        return True
    if re.search(r"^List of (Tables|Figures)", text, re.MULTILINE | re.IGNORECASE):
        if dot_leaders > 2:
            return True
    return False


def is_abbreviations_page(text):
    abbr_count = len(re.findall(r'^[A-Z]{2,}\s', text, re.MULTILINE))
    return abbr_count > 10


def is_skip_page(text):
    if len(text) < 150:
        return True
    if is_toc_page(text):
        return True
    if is_abbreviations_page(text):
        return True
    return False


def count_disco_names(text):
    names = ["Abuja", "Benin", "Eko", "Enugu", "Ibadan", "Ikeja",
             "Jos", "Kaduna", "Kano", "Port Harcourt", "Yola"]
    return sum(1 for n in names if n.lower() in text.lower())


def count_data_lines(text):
    """Count lines with 2+ numbers (integers or decimals), excluding year-like numbers."""
    count = 0
    for line in text.split('\n'):
        nums = re.findall(r'\b\d+\.?\d*\b', line)
        nums = [n for n in nums if not (len(n) == 4 and n.startswith('20'))]
        if len(nums) >= 2:
            count += 1
    return count


def find_table_pages(pdf, markers, min_data_lines=3, max_pages=4,
                     require_disco_names=0, score_fn=None):
    """
    Generic content-based page finder.
    markers: list of regex patterns, ALL must match.
    score_fn: optional function(page_text) -> bonus_score.
    """
    candidates = []
    for pi in range(len(pdf.pages)):
        text = pdf.pages[pi].extract_text() or ""
        if is_skip_page(text):
            continue
        if not all(re.search(pat, text, re.IGNORECASE) for pat in markers):
            continue

        data_lines = count_data_lines(text)
        if data_lines < min_data_lines:
            continue

        if require_disco_names > 0 and count_disco_names(text) < require_disco_names:
            continue

        # Base score
        score = data_lines
        has_table_header = bool(re.search(r'Table\s+\d+', text, re.IGNORECASE))
        is_main_body = pi < len(pdf.pages) * 0.7
        if has_table_header:
            score += 20
        if is_main_body:
            score += 10
        if score_fn:
            score += score_fn(text)

        candidates.append((pi, score))

    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:max_pages]]


def get_multi_page_text(pdf, page_indices, extra_pages=1):
    """Collect text from multiple pages + neighbors, deduplicated and sorted."""
    all_pages = set()
    for pi in page_indices:
        for offset in range(1 + extra_pages):
            candidate = pi + offset
            if 0 <= candidate < len(pdf.pages):
                all_pages.add(candidate)

    texts = []
    for pi in sorted(all_pages):
        text = pdf.pages[pi].extract_text() or ""
        if is_skip_page(text):
            continue
        texts.append(f"--- PAGE {pi + 1} ---\n{text}")
    return "\n\n".join(texts)


# ── Table definitions ──────────────────────────────────────────────────────────

def offtake_score(text):
    """Scoring bonus for offtake pages: prefer MWh/h over ₦ Billion."""
    s = 0
    if re.search(r'MWh/h', text):
        s += 30
    if re.search(r'PCC', text):
        s += 20
    if re.search(r'GWh', text, re.IGNORECASE):
        s += 15
    if re.search(r'₦|Billion|billion', text) and not re.search(r'MWh|GWh', text, re.IGNORECASE):
        s -= 10
    return s


TABLE_DEFS = {
    "paf": {
        "description": "Plant Availability Factor",
        "markers": [
            r"(Egbin|Kainji|Shiroro|Jebba|Delta|Geregu|Afam|Okpai|Omotosho|Sapele|Zungeru|Ihovbor|Odukpani|Omoku|Olorunsogo|Rivers\s*IPP|Calabar|Ughelli|Paras|Gbarain|Trans.?Amadi|Ibom|Ikeja_1|Dadin|AES|ASCO|Azura)",
            r"(availability\s+factor|available\s+capacity|PAF|installed\s+capacity)",
        ],
        "min_data_lines": 3,
        "max_pages": 4,
        "require_disco_names": 0,
        "score_fn": None,
        "prompt": """Extract all Plant Availability Factor data from this text.
Data may be in a table (Table 1) or a bar chart figure (Figure 3, Figure I).
In chart format, numbers are PAF percentages and labels are plant names.

Return ONLY a JSON array. Each element:
- "plant": plant name (string)
- "installed_mw": installed capacity MW (number or null)
- "available_mw_prev": previous quarter available MW (number or null)
- "available_mw_curr": current quarter available MW (number or null)
- "paf_pct_prev": previous quarter PAF % (number or null)
- "paf_pct_curr": current quarter PAF % (number or null)

If only one quarter shown, use _curr fields, _prev = null.
If only PAF % available (no MW), fill paf_pct, leave MW as null.
Skip total/summary/average rows.
Return ONLY the JSON array.""",
    },

    "offtake": {
        "description": "DisCo Energy Offtake",
        "markers": [
            r"(Abuja|Benin|Eko|Enugu|Ibadan|Ikeja|Jos|Kaduna|Kano|Port\s*Harcourt|Yola)",
            r"(MWh/h|PCC|offtake|energy\s+received|energy\s+offtake|GWh)",
        ],
        "min_data_lines": 3,
        "max_pages": 5,
        "require_disco_names": 5,
        "score_fn": offtake_score,
        "prompt": """Extract DisCo energy data from this text. Multiple tables may be present.
I need ENERGY VOLUME tables (GWh or MWh/h), NOT billing/revenue tables (₦ Billion).

Priority: (1) Table with MWh/h and PCC columns. (2) Table with GWh columns. (3) Skip ₦-only tables.

Return ONLY a JSON array. Each element:
- "disco": DisCo name (string)
Then include whichever columns exist:
- "energy_offtake_mwh_h_prev", "energy_offtake_mwh_h_curr"
- "pcc_mwh_h_prev", "pcc_mwh_h_curr"
- "offtake_pct_prev", "offtake_pct_curr"
- "energy_received_gwh_prev", "energy_received_gwh_curr"
- "energy_billed_gwh_prev", "energy_billed_gwh_curr"
- "billing_efficiency_pct_prev", "billing_efficiency_pct_curr"

Use null for fields not present. Skip total/summary rows.
Return ONLY the JSON array.""",
    },

    "atcc": {
        "description": "ATC&C Losses",
        "markers": [
            r"(Abuja|Benin|Eko|Enugu|Ibadan|Ikeja|Jos|Kaduna|Kano|Port\s*Harcourt|Yola)",
            r"(ATC&?C|aggregate\s+technical|commercial\s+and\s+collection)",
        ],
        "min_data_lines": 3,
        "max_pages": 3,
        "require_disco_names": 5,
        "score_fn": None,
        "prompt": """Extract the ATC&C (Aggregate Technical, Commercial and Collection) loss table.

Return ONLY a JSON array. Each element:
- "disco": DisCo name (string)
- "myto_target_pct": MYTO target ATC&C % (number or null)
- "atcc_pct_prev": previous quarter ATC&C % (number or null)
- "atcc_pct_curr": current quarter ATC&C % (number or null)

If only one quarter shown, use _curr, set _prev = null.
Skip total/summary rows.
Return ONLY the JSON array.""",
    },

    "metering": {
        "description": "Metering Progress",
        "markers": [
            r"(Abuja|Benin|Eko|Enugu|Ibadan|Ikeja|Jos|Kaduna|Kano|Port\s*Harcourt|Yola)",
            r"(meter(ed|ing)|metered\s+customers|metering\s+rate|metering\s+gap|registered\s+customer)",
        ],
        "min_data_lines": 3,
        "max_pages": 3,
        "require_disco_names": 5,
        "score_fn": None,
        "prompt": """Extract the metering progress table.

Return ONLY a JSON array. Each element:
- "disco": DisCo name (string)
- "total_registered_customers": total registered customers (number or null)
- "metered_customers": number of metered customers (number or null)
- "metering_rate_pct": metering rate % (number or null)

Use the most recent snapshot values (not quarter-over-quarter change).
Skip total/summary rows.
Return ONLY the JSON array.""",
    },
}


# ── Claude extraction ──────────────────────────────────────────────────────────

def extract_with_claude(client, page_text, prompt):
    """Send page text to Claude, return (data_list, error_string)."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": f"""Here is text from a Nigerian Electricity Regulatory Commission (NERC) quarterly report.
Multiple pages included — find the relevant table data across all pages.

{page_text}

{prompt}"""
            }],
        )

        raw = ""
        for block in response.content:
            block_text = getattr(block, "text", None)
            if block_text and block_text.strip():
                candidate = block_text.strip()
                if candidate.startswith("[") or candidate.startswith("{"):
                    raw = candidate
                    break
                if not raw:
                    raw = candidate

        if not raw:
            block_types = [getattr(b, "type", "?") for b in response.content]
            return None, f"No text in response. Types: {block_types}"

        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        if not raw.startswith("["):
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                raw = match.group(0)

        data = json.loads(raw)
        if not isinstance(data, list):
            return None, f"Expected list, got {type(data).__name__}"
        return data, None

    except json.JSONDecodeError as e:
        return None, f"JSON error: {e} — raw: {raw[:300]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ── Process one report ─────────────────────────────────────────────────────────

def process_report(pdf_path, client):
    """Extract all tables from one NERC quarterly PDF. Returns a dict."""
    filename = os.path.basename(pdf_path)
    pdf = pdfplumber.open(pdf_path)
    year, quarter = detect_period(filename, pdf)

    report = {
        "file": filename,
        "year": year,
        "quarter": quarter,
        "period": f"{year}Q{quarter}" if year and quarter else "UNKNOWN",
        "total_pages": len(pdf.pages),
        "extracted_at": datetime.now().isoformat(),
        "tables": {},
    }

    for table_key, tdef in TABLE_DEFS.items():
        entry = {"pages": None, "row_count": 0, "data": None, "error": None}

        pages = find_table_pages(
            pdf,
            markers=tdef["markers"],
            min_data_lines=tdef["min_data_lines"],
            max_pages=tdef["max_pages"],
            require_disco_names=tdef.get("require_disco_names", 0),
            score_fn=tdef.get("score_fn"),
        )

        if not pages:
            entry["error"] = "No matching pages found"
            report["tables"][table_key] = entry
            continue

        entry["pages"] = [p + 1 for p in pages]
        page_text = get_multi_page_text(pdf, pages, extra_pages=1)

        data, error = extract_with_claude(client, page_text, tdef["prompt"])
        if data:
            entry["data"] = data
            entry["row_count"] = len(data)
        else:
            entry["error"] = error

        report["tables"][table_key] = entry

        # Brief pause to avoid rate limits
        time.sleep(0.5)

    pdf.close()
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def load_existing(path):
    """Load previously extracted reports for resume support."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    # Validate setup
    if not os.path.isdir(REPORTS_DIR):
        print(f"ERROR: Reports directory not found: {REPORTS_DIR}")
        sys.exit(1)

    pdfs = get_quarterly_pdfs(REPORTS_DIR)
    print(f"Found {len(pdfs)} quarterly report PDFs\n")

    if dry_run:
        for f in pdfs:
            y, q = detect_period_from_filename(f)
            tag = f"{y}Q{q}" if y else "???"
            print(f"  {tag:8s}  {f}")
        print(f"\nDry run complete. {len(pdfs)} files to process.")
        sys.exit(0)

    # API client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: ANTHROPIC_API_KEY=sk-... python batch_extract.py")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Resume support: load existing results
    results = load_existing(OUTPUT_FILE)
    done_files = {r["file"] for r in results}
    remaining = [f for f in pdfs if f not in done_files]

    if done_files:
        print(f"Resuming: {len(done_files)} already done, {len(remaining)} remaining\n")
    else:
        print(f"Processing {len(remaining)} files\n")

    for i, filename in enumerate(remaining, 1):
        pdf_path = os.path.join(REPORTS_DIR, filename)
        print(f"[{i}/{len(remaining)}] {filename}")

        try:
            report = process_report(pdf_path, client)

            # Print summary
            for tk, td in report["tables"].items():
                status = f"✅ {td['row_count']}" if td["data"] else f"❌ {td['error'][:60] if td['error'] else '?'}"
                pages = td.get("pages", "—")
                print(f"    {tk:10s} {status:30s} pages={pages}")

            results.append(report)

            # Save after each report (crash-safe)
            save_results(results, OUTPUT_FILE)

        except Exception as e:
            print(f"    ⚠️  FAILED: {type(e).__name__}: {e}")
            results.append({
                "file": filename,
                "year": None, "quarter": None, "period": "ERROR",
                "total_pages": 0,
                "extracted_at": datetime.now().isoformat(),
                "tables": {},
                "error": str(e),
            })
            save_results(results, OUTPUT_FILE)

        print()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Extraction complete: {len(results)} reports in {OUTPUT_FILE}")
    ok = sum(1 for r in results if any(
        r.get("tables", {}).get(t, {}).get("data") for t in TABLE_DEFS
    ))
    print(f"Reports with at least one table extracted: {ok}/{len(results)}")