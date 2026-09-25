"""
NERC Retry Extraction — Target specific failed file+table combos
=================================================================
Reads the failure map, retries only what's needed with improved
page finding and prompts for problematic formats.

Output: nerc_raw_extract_retry.json
"""

import pdfplumber
import anthropic
import re
import os
import sys
import json
import time
from datetime import datetime

REPORTS_DIR = "Disequilibrium/NERC REPORTS"
OUTPUT_FILE = "nerc_raw_extract_retry.json"
MODEL = "claude-sonnet-4-6"

# ── Exact retry manifest ───────────────────────────────────────────────────────
# Each entry: (filename, [list of tables to retry])

RETRY_MANIFEST = [
    # Category A: Connection errors — just retry
    ("2025_Q1_Report.pdf",            ["offtake", "atcc", "metering"]),
    ("2025_Q2-Report.pdf",            ["paf", "offtake", "atcc", "metering"]),
    ("2025_Q3-Report.pdf",            ["paf", "offtake", "atcc", "metering"]),
    ("2025_Q4-Report.pdf",            ["paf", "offtake", "atcc", "metering"]),

    # Category B: PAF empty extraction — pages found, Claude returned []
    ("2NERC_2019Q1 Report_REV.pdf",   ["paf"]),
    ("2019Q3_FINAL.pdf",              ["paf"]),
    ("2019Q4_Final_Online_.pdf",      ["paf"]),
    ("NERC Second Quarter Report 2020.pdf", ["paf", "metering"]),
    ("NERC Quarterly Report_Q3 2020.pdf",   ["paf", "metering"]),
    ("NERC Quarterly Report_Q4 2020.pdf",   ["paf"]),
    ("2021_Q3_REPORT.pdf",            ["paf"]),
    ("NERCSecondQuarterReport2023.pdf", ["paf"]),

    # Category C: PAF no matching pages — old format / scanner too strict
    ("2017Q4_UPDATED_FINAL_.pdf",     ["paf"]),
    ("NERC Second Quarter Report 2018.pdf", ["paf"]),
    ("NERC QUARTERLY REPORT_Q1 2021.pdf",   ["paf"]),
    ("NERC QUARTERLY REPORT_ Q2 2021.pdf",  ["paf"]),

    # Category D: Total failure — possibly image PDFs
    ("NERC_2018Q4 Report.pdf",        ["paf", "offtake", "atcc", "metering"]),
    ("Quarterly Reports_2020Q1_FINAL.pdf", ["paf", "offtake", "atcc", "metering"]),

    # Category E: Metering empty
    ("NERCThirdQuarter2023Report.pdf", ["metering"]),
]

# ── Period detection (same as batch_extract.py) ────────────────────────────────

QUARTER_WORDS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4}

def detect_period_from_filename(filename):
    name = filename.upper().replace("-", " ").replace("_", " ")
    m = re.search(r'(\d{4})\s*Q(\d)', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'Q(\d)\s*(\d{4})', name)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r'(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\s+(\d{4})', name)
    if m:
        return int(m.group(2)), QUARTER_WORDS[m.group(1)]
    for word, qnum in QUARTER_WORDS.items():
        if word in name:
            year_match = re.search(r'(20\d{2})', name)
            if year_match:
                return int(year_match.group(1)), qnum
    return None, None

def detect_period_from_pdf(pdf):
    for i in range(min(5, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ""
        m = re.search(r"(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\s+(20\d{2})", text, re.IGNORECASE)
        if m:
            return int(m.group(2)), QUARTER_WORDS[m.group(1).upper()]
    return None, None

def detect_period(filename, pdf):
    year, quarter = detect_period_from_filename(filename)
    if year and quarter:
        return year, quarter
    return detect_period_from_pdf(pdf)


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
    return len(re.findall(r'^[A-Z]{2,}\s', text, re.MULTILINE)) > 10

def is_skip_page(text):
    return len(text) < 100 or is_toc_page(text) or is_abbreviations_page(text)

def count_disco_names(text):
    names = ["Abuja", "Benin", "Eko", "Enugu", "Ibadan", "Ikeja",
             "Jos", "Kaduna", "Kano", "Port Harcourt", "Yola"]
    return sum(1 for n in names if n.lower() in text.lower())

def count_data_lines(text):
    count = 0
    for line in text.split('\n'):
        nums = re.findall(r'\b\d+\.?\d*\b', line)
        nums = [n for n in nums if not (len(n) == 4 and n.startswith('20'))]
        if len(nums) >= 2:
            count += 1
    return count

def get_multi_page_text(pdf, page_indices, extra_pages=1):
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

def check_if_image_pdf(pdf, sample_pages=5):
    """Check if PDF is scanned images (no extractable text)."""
    total_text = 0
    for i in range(min(sample_pages, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ""
        total_text += len(text.strip())
    return total_text < 200  # Less than 200 chars across 5 pages = image PDF


# ── Table-specific page finders ────────────────────────────────────────────────

def find_paf_pages(pdf):
    """
    Broadened PAF page finder. Handles:
    - Tabular format (Table 1) with decimal values
    - Bar chart format (Figure 3) with integer values and ALL CAPS names
    - Appendix figures (Figure I) with comparison data
    Looser thresholds than the batch script.
    """
    # Plant name pattern — very broad, case insensitive
    plant_pat = re.compile(
        r"(Egbin|Kainji|Shiroro|Jebba|Delta|Geregu|Afam|Okpai|Omotosho|"
        r"Sapele|Zungeru|Ihovbor|Odukpani|Omoku|Olorunsogo|Rivers|"
        r"Calabar|Ughelli|Paras|Gbarain|Trans.?Amadi|Ibom|Ikeja|Dadin|"
        r"AES|ASCO|Azura|Dadinkowa|NIPP|Olorunsogo)",
        re.IGNORECASE
    )

    # Context pattern — very broad
    context_pat = re.compile(
        r"(availability|available|PAF|capacity|factor|generation\s+plant|"
        r"installed|MW\b|average\s+availability)",
        re.IGNORECASE
    )

    candidates = []
    for pi in range(len(pdf.pages)):
        text = pdf.pages[pi].extract_text() or ""
        if is_skip_page(text):
            continue

        # Count plant name matches
        plant_matches = len(set(m.group().upper() for m in plant_pat.finditer(text)))
        if plant_matches < 3:
            continue

        # Must have some context
        if not context_pat.search(text):
            continue

        # Count any numbers (integers or decimals)
        any_nums = len(re.findall(r'\b\d+\.?\d*\b', text))
        # Filter years
        non_year_nums = len([n for n in re.findall(r'\b\d+\.?\d*\b', text)
                            if not (len(n) == 4 and n.startswith('20'))])

        score = plant_matches * 5 + non_year_nums
        has_table = bool(re.search(r'Table\s+\d+', text, re.IGNORECASE))
        is_main = pi < len(pdf.pages) * 0.7
        has_figure = bool(re.search(r'Figure\s+\d+', text, re.IGNORECASE))

        if has_table:
            score += 30
        if is_main:
            score += 10
        if has_figure:
            score += 5

        candidates.append((pi, score))

    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:4]]


def find_generic_pages(pdf, markers, min_data_lines=3, max_pages=4,
                       require_disco_names=0, score_fn=None):
    """Same generic finder as batch_extract.py."""
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
        score = data_lines
        if re.search(r'Table\s+\d+', text, re.IGNORECASE):
            score += 20
        if pi < len(pdf.pages) * 0.7:
            score += 10
        if score_fn:
            score += score_fn(text)
        candidates.append((pi, score))
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:max_pages]]


def offtake_score(text):
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


# ── Table definitions ──────────────────────────────────────────────────────────

TABLE_DEFS = {
    "paf": {
        "find_fn": "paf",  # use find_paf_pages
        "prompt": """Extract ALL Plant Availability Factor data from this text.

The data may appear as:
- A formal table (Table 1) with columns for installed capacity, available capacity, and PAF %
- A bar chart (Figure 3 or Figure I) where numbers are PAF percentages next to plant names
- A list of plants with percentage values

IMPORTANT: Even if the data comes from a chart/figure rather than a formal table, extract every plant and its PAF percentage.

Return ONLY a JSON array. Each element:
- "plant": plant name (string)
- "installed_mw": installed capacity MW (number or null)
- "available_mw_prev": previous quarter available MW (number or null)
- "available_mw_curr": current quarter available MW (number or null)
- "paf_pct_prev": previous quarter PAF % (number or null)
- "paf_pct_curr": current quarter PAF % (number or null)

If only one quarter shown, use _curr fields, _prev = null.
If only PAF % available (no MW), fill paf_pct, leave MW as null.
Extract EVERY plant listed, even if some values are 0.
Skip total/summary/average rows.
Return ONLY the JSON array.""",
    },

    "offtake": {
        "find_fn": "generic",
        "markers": [
            r"(Abuja|Benin|Eko|Enugu|Ibadan|Ikeja|Jos|Kaduna|Kano|Port\s*Harcourt|Yola)",
            r"(MWh/h|PCC|offtake|energy\s+received|energy\s+offtake|GWh)",
        ],
        "min_data_lines": 3,
        "max_pages": 5,
        "require_disco_names": 5,
        "score_fn": offtake_score,
        "prompt": """Extract DisCo energy data. Multiple tables may be present.
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
        "find_fn": "generic",
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
        "find_fn": "generic",
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
        if len(data) == 0:
            return None, "Claude returned empty array []"
        return data, None

    except json.JSONDecodeError as e:
        return None, f"JSON error: {e} — raw: {raw[:300]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ── Process one table from one report ──────────────────────────────────────────

def extract_table(pdf, table_key, client):
    """Find pages and extract one table. Returns (data, pages, error)."""
    tdef = TABLE_DEFS[table_key]

    # Find pages
    if tdef["find_fn"] == "paf":
        pages = find_paf_pages(pdf)
    else:
        pages = find_generic_pages(
            pdf,
            markers=tdef["markers"],
            min_data_lines=tdef.get("min_data_lines", 3),
            max_pages=tdef.get("max_pages", 4),
            require_disco_names=tdef.get("require_disco_names", 0),
            score_fn=tdef.get("score_fn"),
        )

    if not pages:
        return None, None, "No matching pages found"

    page_text = get_multi_page_text(pdf, pages, extra_pages=1)

    if len(page_text.strip()) < 100:
        return None, [p + 1 for p in pages], "Page text too short (possible image PDF)"

    data, error = extract_with_claude(client, page_text, tdef["prompt"])
    return data, [p + 1 for p in pages], error


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Load existing retry results for resume
    results = []
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
        for r in results:
            for tk in r.get("tables", {}):
                td = r["tables"][tk]
                if td.get("data") and len(td["data"]) > 0:
                    done_keys.add((r["file"], tk))
        print(f"Loaded {len(results)} existing retry results\n")

    # Build retry list excluding already-done combos
    retry_list = []
    for filename, table_keys in RETRY_MANIFEST:
        remaining = [tk for tk in table_keys if (filename, tk) not in done_keys]
        if remaining:
            retry_list.append((filename, remaining))

    total_calls = sum(len(tks) for _, tks in retry_list)
    print(f"Retry manifest: {len(retry_list)} files, {total_calls} table extractions\n")

    for i, (filename, table_keys) in enumerate(retry_list, 1):
        pdf_path = os.path.join(REPORTS_DIR, filename)
        if not os.path.exists(pdf_path):
            print(f"[{i}/{len(retry_list)}] MISSING: {filename}")
            continue

        print(f"[{i}/{len(retry_list)}] {filename}")

        pdf = pdfplumber.open(pdf_path)
        year, quarter = detect_period(filename, pdf)

        # Check for image PDF
        if check_if_image_pdf(pdf):
            print(f"    ⚠️  IMAGE PDF — no extractable text, skipping all tables")
            entry = {
                "file": filename,
                "year": year, "quarter": quarter,
                "period": f"{year}Q{quarter}" if year and quarter else "UNKNOWN",
                "total_pages": len(pdf.pages),
                "extracted_at": datetime.now().isoformat(),
                "tables": {tk: {"pages": None, "row_count": 0, "data": None,
                                "error": "Image PDF — no extractable text"} for tk in table_keys},
                "is_image_pdf": True,
            }
            results.append(entry)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            pdf.close()
            print()
            continue

        # Find or create entry for this file
        entry = None
        for r in results:
            if r["file"] == filename:
                entry = r
                break
        if not entry:
            entry = {
                "file": filename,
                "year": year, "quarter": quarter,
                "period": f"{year}Q{quarter}" if year and quarter else "UNKNOWN",
                "total_pages": len(pdf.pages),
                "extracted_at": datetime.now().isoformat(),
                "tables": {},
            }
            results.append(entry)

        for tk in table_keys:
            data, pages, error = extract_table(pdf, tk, client)
            td = {"pages": pages, "row_count": 0, "data": None, "error": error}
            if data:
                td["data"] = data
                td["row_count"] = len(data)
                td["error"] = None
            entry["tables"][tk] = td

            status = f"✅ {td['row_count']}" if data else f"❌ {error[:60] if error else '?'}"
            print(f"    {tk:10s} {status:40s} pages={pages}")

            # Save after each extraction
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(0.5)

        pdf.close()
        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"Retry complete: {OUTPUT_FILE}")
    ok = sum(1 for r in results if any(
        r.get("tables", {}).get(t, {}).get("data") for t in TABLE_DEFS
    ))
    print(f"Reports with data: {ok}/{len(results)}")

    # Show remaining gaps
    print("\nRemaining gaps:")
    for r in sorted(results, key=lambda x: x.get("period", "")):
        for tk in TABLE_DEFS:
            td = r.get("tables", {}).get(tk, {})
            if not (td.get("data") and len(td.get("data", [])) > 0):
                err = (td.get("error") or "not attempted")[:50]
                print(f"  {r.get('period','?'):8s} {tk:10s} {err}")