import os
import re
import json
import sqlite3
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ---------------- Config & Data Paths  ----------------
RAIN_FILE = "Rainfall.csv"
AGRI_FILE = "Crop.csv"
DB_FILE = "database.db"

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------------- Canonicalization & Mappings ----------------
ALIASES = {
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",
    "uttaranchal": "uttarakhand",
    "jammu & kashmir": "jammu and kashmir",
    "jammu and kashmir": "jammu and kashmir",
    "dadra and nagar haveli and daman and diu": "daman and diu",
}

def canon_state(s: str) -> str:
    if s is None:
        return ""
    x = str(s)
    x = re.sub(r"\(.*?\)", "", x)
    x = x.replace("&", "and")
    x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    x = ALIASES.get(x, x)
    return x

def pretty_state(s: str) -> str:
    return str(s).strip().title()

# ---------------- Subdivision-to-State Mapping  ----------------
SUB_TO_STATES = {
    "COASTAL ANDHRA PRADESH": ["Andhra Pradesh"],
    "RAYALSEEMA": ["Andhra Pradesh"],
    "TELANGANA": ["Telangana"],

    "KONKAN & GOA": ["Maharashtra", "Goa"],
    "NORTH INTERIOR KARNATAKA": ["Karnataka"],
    "SOUTH INTERIOR KARNATAKA": ["Karnataka"],
    "COASTAL KARNATAKA": ["Karnataka"],

    "SUB HIMALAYAN WEST BENGAL & SIKKIM": ["West Bengal", "Sikkim"],
    "GANGETIC WEST BENGAL": ["West Bengal"],
    "ASSAM & MEGHALAYA": ["Assam", "Meghalaya"],
    "NAGALAND, MANIPUR, MIZORAM, TRIPURA": ["Nagaland", "Manipur", "Mizoram", "Tripura"],
    "HARYANA, CHANDIGARH & DELHI": ["Haryana", "Chandigarh", "Delhi"],

    "EAST RAJASTHAN": ["Rajasthan"],
    "WEST RAJASTHAN": ["Rajasthan"],
    "EAST MADHYA PRADESH": ["Madhya Pradesh"],
    "WEST MADHYA PRADESH": ["Madhya Pradesh"],
    "EAST UTTAR PRADESH": ["Uttar Pradesh"],
    "WEST UTTAR PRADESH": ["Uttar Pradesh"],
    "GUJARAT REGION": ["Gujarat"],
    "SAURASHTRA & KUTCH": ["Gujarat"],

    "BIHAR": ["Bihar"],
    "JHARKHAND": ["Jharkhand"],
    "ODISHA": ["Odisha"],
    "CHHATTISGARH": ["Chhattisgarh"],
    "HIMACHAL PRADESH": ["Himachal Pradesh"],
    "JAMMU & KASHMIR": ["Jammu and Kashmir"],
    "PUNJAB": ["Punjab"],
    "UTTARAKHAND": ["Uttarakhand"],
    "TAMIL NADU": ["Tamil Nadu"],
    "KERALA": ["Kerala"],
    "ARUNACHAL PRADESH": ["Arunachal Pradesh"],
    "ANDAMAN & NICOBAR ISLANDS": ["Andaman and Nicobar Islands"],
    "LAKSHADWEEP": ["Lakshadweep"],
}

def norm_sub(s: str) -> str:
    return str(s).strip().upper().replace("-", " ").replace("  ", " ")

# ---------------- Crop normalization  ----------------
CROP_PATTERNS = [
    (r"(?i)\brice\b", "Rice", 1000),
    (r"(?i)\bwheat\b", "Wheat", 1000),
    (r"(?i)\bmaize\b", "Maize", 1000),
    (r"(?i)\bjowar\b", "Jowar", 1000),
    (r"(?i)\bbajra\b", "Bajra", 1000),
    (r"(?i)\bragi\b", "Ragi", 1000),
    (r"(?i)\bbarley\b", "Barley", 1000),
    (r"(?i)\bsmall millets\b", "Small millets", 1000),
    (r"(?i)\bgram\b", "Gram", 1000),
    (r"(?i)\btur\b", "Tur", 1000),
    (r"(?i)\bother pulses\b", "Other pulses", 1000),
    (r"(?i)\bgroundnuts?\b", "Groundnut", 1000),
    (r"(?i)\bcastor ?seed\b|castorseed", "Castor seed", 1000),
    (r"(?i)\blinseed\b", "Linseed", 1000),
    (r"(?i)\bsesamum\b", "Sesamum", 1000),
    (r"(?i)\brapeseed.*mustard\b|rapeseed and", "Rapeseed & Mustard", 1000),
    (r"(?i)\btotal cereals\b", "Total cereals", 1000),
    (r"(?i)\btotal pulses\b", "Total pulses", 1000),
    (r"(?i)\btotal food grains\b", "Total Food Grains", 1000),
    (r"(?i)\btotal oilseeds\*?\b", "Total Oilseeds", 1000),
    # Sugarcane variants
    (r"(?i)\bsugarcane\b.*\(000mt\)", "Sugarcane", 1000),
    (r"(?i)\bsugarcane\b.*\(th\.? tonnes\)", "Sugarcane", 1000),
    (r"(?i)\bsugarcane\b(?!.*\()", "Sugarcane", 1000),
]

def clean_crop_label(raw: str):
    s = str(raw)
    for pat, clean, scale in CROP_PATTERNS:
        if re.search(pat, s):
            return clean, scale
    return s, 1000  # fallback assume thousand-tonne style

def parse_crop_year_helper(s: str):
    s = str(s)
    m = re.match(r"(.+?)-(\d{4})-(\d{2})", s)  # e.g., "Rice-2013-14"
    if m:
        crop = m.group(1).strip()
        year = int(m.group(2))
        return pd.Series([crop, year])
    return pd.Series([None, None])

# ---------------- DATABASE ETL Extract, Transform, and Load ----------------
def ensure_db_exists():
    """Runs the ETL process only if the database file does not exist."""
    if os.path.exists(DB_FILE):
        return

    print(f"--- Database '{DB_FILE}' not found. Running one-time data processing (ETL)... ---")
    
    # ---------------- Checks ----------------
    if not os.path.exists(RAIN_FILE):
        raise FileNotFoundError(RAIN_FILE)
    if not os.path.exists(AGRI_FILE):
        raise FileNotFoundError(AGRI_FILE)

    # ---------------- RAINFALL ----------------
    print("--- Processing Rainfall ---")
    rf = pd.read_csv(RAIN_FILE)
    need_cols_rf = {"SUBDIVISION", "YEAR", "ANNUAL"}
    miss = need_cols_rf - set(rf.columns)
    if miss:
        raise ValueError(f"Rainfall CSV missing columns: {sorted(miss)}")

    rows = []
    for _, row in rf.iterrows():
        sub = norm_sub(row["SUBDIVISION"])
        yr = int(row["YEAR"])
        try:
            val = float(row["ANNUAL"])
        except Exception:
            continue
        states = SUB_TO_STATES.get(sub, [str(row["SUBDIVISION"]).strip().title()])
        for st in states:
            st_clean = st.strip()
            rows.append({
                "State": st_clean,
                "StateNorm": canon_state(st_clean),
                "Year": yr,
                "Annual_Rainfall": val
            })
    rf_df = pd.DataFrame(rows)
    rf_final = (
        rf_df.groupby(["State", "StateNorm", "Year"], as_index=False)["Annual_Rainfall"]
             .mean()
    )

    # ---------------- AGRICULTURE ----------------
    print("\n--- Processing Agriculture ---")
    ag = pd.read_csv(AGRI_FILE)
    if "State/ UT Name" not in ag.columns:
        raise ValueError("Missing 'State/ UT Name' in agriculture CSV")

    id_col = ["State/ UT Name"]
    value_cols = [c for c in ag.columns if c not in id_col]
    long_ag = pd.melt(
        ag, id_vars=id_col, value_vars=value_cols,
        var_name="Crop_Year_Raw", value_name="Production"
    )

    long_ag[["Crop", "Year"]] = long_ag["Crop_Year_Raw"].apply(parse_crop_year_helper)
    long_ag = long_ag.dropna(subset=["Crop", "Year"])
    long_ag["Production"] = pd.to_numeric(long_ag["Production"], errors="coerce")
    long_ag = long_ag.dropna(subset=["Production"])

    long_ag["State"] = long_ag["State/ UT Name"].astype(str).str.strip().str.title()
    long_ag["StateNorm"] = long_ag["State"].apply(canon_state)

    # crop normalization -> MT
    long_ag["CropSimple"], long_ag["UnitScale"] = zip(*long_ag["Crop"].map(clean_crop_label))
    long_ag["ProductionMT"] = long_ag["Production"] * long_ag["UnitScale"]

    final_ag = long_ag[["State", "StateNorm", "Year", "Crop", "CropSimple", "Production", "ProductionMT"]]

    # ---------------- Write DB ----------------
    with sqlite3.connect(DB_FILE) as conn:
        rf_final.to_sql("rainfall", conn, if_exists="replace", index=False)
        final_ag.to_sql("crop_production", conn, if_exists="replace", index=False)

    print(f"✓ Rainfall table created with {len(rf_final)} rows.")
    print(f"✓ Agriculture table created with {len(final_ag)} rows.")
    print(f"\nAll done! Database '{DB_FILE}' ready.")


# ---------------- Groq Client Initialization ----------------
ensure_db_exists()

client = None
if API_KEY:
    try:
        client = Groq(api_key=API_KEY)
        print(f"--- Using Groq Model: {MODEL_NAME} ---")
    except Exception as e:
        # keep app running; log warning for operator
        print(f"Warning: could not initialize Groq client: {e}")
else:
    print("Warning: GROQ_API_KEY not found in .env — chatbot will be disabled until configured.")

# ---------------- Helpers: available agri years ----------------
def parse_args(raw: str) -> dict:
    try:
        return json.loads(raw)
    except:
        try:
            return json.loads(raw.replace("'", '"'))
        except:
            return {}

def agri_year_bounds():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT MIN(Year) AS miny, MAX(Year) AS maxy FROM crop_production;", conn)
    return int(df.loc[0,"miny"]), int(df.loc[0,"maxy"])

# ---------------- Rainfall ----------------
def get_rainfall_data(states: list[str], years: list[int]) -> str:
    states = [pretty_state(s) for s in states]
    norms = [canon_state(s) for s in states]
    print(f"--- TOOL: get_rainfall_data states={states}, years={years} ---")
    try:
        conn = sqlite3.connect(DB_FILE)
        ph_s = ",".join("?" * len(norms))
        ph_y = ",".join("?" * len(years))
        q = f"""
            SELECT State, StateNorm, Year, Annual_Rainfall
            FROM rainfall
            WHERE StateNorm IN ({ph_s})
              AND Year IN ({ph_y})
            ORDER BY State, Year;
        """
        df = pd.read_sql_query(q, conn, params=tuple(norms + years))
        conn.close()

        if df.empty:
            return f"No rainfall data found for {states} in {years}."

        lines = ["Rainfall", "--------"]
        for i, st in enumerate(states):
            stn = norms[i]
            sub = df[df["StateNorm"] == stn]
            if sub.empty:
                lines.append(f"• {st}: No rainfall data for {years}.")
            else:
                disp = sub["State"].iloc[0]
                lines.append(f"• {disp}")
                for _, r in sub.iterrows():
                    lines.append(f"   - {int(r['Year'])}: {float(r['Annual_Rainfall']):.1f} mm")
        lines.append("[Source: IMD Area Weighted Annual Rainfall (1901–2015)]")
        return "\n".join(lines)

    except Exception as e:
        return f"Rainfall error: {e}"

# ---------------- Crop: Top N (MT) with year fallback ----------------
def get_top_crops(states: list[str], years: list[int], top_m: int) -> str:
    states = [pretty_state(s) for s in states]
    norms = [canon_state(s) for s in states]
    print(f"--- TOOL: get_top_crops states={states}, years={years}, top_m={top_m} ---")
    try:
        miny, maxy = agri_year_bounds()  # e.g., 2009–2015 for your file
        # Clip years to available range; if nothing remains, pick nearest single year
        usable_years = [y for y in years if miny <= y <= maxy]
        note = ""
        if not usable_years:
            # choose nearest available year to the requested median
            target = sorted(years)[len(years)//2]
            nearest = max(min(target, maxy), miny)
            usable_years = [nearest]
            note = f"(No crop data for {years}; showing {nearest} instead.)"

        conn = sqlite3.connect(DB_FILE)
        summaries = []
        for i, st in enumerate(states):
            stn = norms[i]
            ph = ",".join("?" * len(usable_years))
            q = f"""
                SELECT CropSimple AS Crop, SUM(ProductionMT) AS Total_Production
                FROM crop_production
                WHERE StateNorm = ?
                  AND Year IN ({ph})
                GROUP BY CropSimple
                ORDER BY Total_Production DESC
                LIMIT ?
            """
            df = pd.read_sql_query(q, conn, params=(stn, *usable_years, top_m))
            if df.empty:
                summaries.append(f"No crop data was available for {st}.")
            else:
                items = [f"{idx+1}. {row['Crop']} ({row['Total_Production']:,.0f} MT)"
                         for idx, row in df.iterrows()]
                span = f"{min(usable_years)}–{max(usable_years)}" if len(usable_years)>1 else f"{usable_years[0]}"
                summaries.append(f"The top {top_m} crops in {st} for {span} were: " + " ".join(items))
        conn.close()

        header = "Top Crops\n---------\n"
        if note:
            header += note + "\n"
        return header + "\n".join(summaries) + "\n[Source: State/UT-wise Production of Principal Crops (2009–2015)]"

    except Exception as e:
        return f"Crops error: {e}"

# ---------------- Crop: Total sum (MT)  ----------------
def get_total_crop_production(states: list[str], years: list[int]) -> str:
    states = [pretty_state(s) for s in states]
    norms = [canon_state(s) for s in states]
    print(f"--- TOOL: get_total_crop_production states={states}, years={years} ---")
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = []
        for i, st in enumerate(states):
            stn = norms[i]
            ph = ",".join("?" * len(years))
            q = f"""
                SELECT SUM(ProductionMT) AS Total
                FROM crop_production
                WHERE StateNorm = ?
                  AND Year IN ({ph});
            """
            df = pd.read_sql_query(q, conn, params=(stn, *years))
            total = float(df.loc[0, "Total"]) if not df.empty and pd.notna(df.loc[0, "Total"]) else 0.0
            rows.append((st, total))
        conn.close()

        lines = ["Total Crop Production", "---------------------"]
        if len(years) == 1:
            for (st, total) in rows:
                lines.append(f"• {st}: {total:,.0f} MT (total in {years[0]})")
        else:
            span = f"{min(years)}–{max(years)}"
            for (st, total) in rows:
                lines.append(f"• {st}: {total:,.0f} MT (total across {span})")

        lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
        return "\n".join(lines)

    except Exception as e:
        return f"Total crops error: {e}"

# ---------------- NEW: Rainfall vs Total Crop correlation (Pearson r) over last N years ----------------
def get_rainfall_crop_correlation(state: str, last_n: int) -> str:
    state_disp = pretty_state(state)
    stn = canon_state(state)
    print(f"--- TOOL: get_rainfall_crop_correlation state={state_disp}, last_n={last_n} ---")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # latest common year available in both tables for the state
            y_r = pd.read_sql_query("SELECT MAX(Year) AS y FROM rainfall WHERE StateNorm = ?", conn, params=(stn,))
            y_c = pd.read_sql_query("SELECT MAX(Year) AS y FROM crop_production WHERE StateNorm = ?", conn, params=(stn,))
            if y_r.empty or pd.isna(y_r.loc[0,"y"]) or y_c.empty or pd.isna(y_c.loc[0,"y"]):
                return f"No data available for {state_disp}."

            ymax = int(min(y_r.loc[0,"y"], y_c.loc[0,"y"]))
            ymin = ymax - int(last_n) + 1

            rain = pd.read_sql_query("""
                SELECT Year, Annual_Rainfall AS rain
                FROM rainfall
                WHERE StateNorm = ? AND Year BETWEEN ? AND ?
                ORDER BY Year
            """, conn, params=(stn, ymin, ymax))

            crops = pd.read_sql_query("""
                SELECT Year, SUM(ProductionMT) AS prod
                FROM crop_production
                WHERE StateNorm = ? AND Year BETWEEN ? AND ?
                GROUP BY Year
                ORDER BY Year
            """, conn, params=(stn, ymin, ymax))

        df = pd.merge(rain, crops, on="Year", how="inner").dropna()
        if len(df) < 3:
            return f"Insufficient overlapping years for {state_disp} between {ymin}–{ymax}."

        r = float(df["rain"].corr(df["prod"]))

        def interp(val):
            a = abs(val)
            if a >= 0.7:  return "strong"
            if a >= 0.4:  return "moderate"
            if a >= 0.2:  return "weak"
            return "very weak/none"

        sign = "positive" if r >= 0 else "negative"
        lines = [
            f"Rainfall ↔ Crop Production Correlation — {state_disp}",
            "-----------------------------------------------",
            f"Years used: {df['Year'].min()}–{df['Year'].max()} (n={len(df)})",
            # CHANGE: Increased precision to 3 decimal places for "exact results"
            f"Pearson r = {r:.3f} ({interp(r)} {sign} relationship)",
            "[Source: IMD Area Weighted Annual Rainfall (1901–2015)]",
            "[Source: State/UT-wise Production of Principal Crops (2009–2015)]",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Correlation error: {e}"

# ---------------- Tool List ----------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_rainfall_data",
            "description": "Get rainfall (mm) for given states and years (exact years).",
            "parameters": {
                "type": "object",
                "properties": {
                    "states": {"type": "array", "items": {"type": "string"}},
                    "years": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["states", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_crops",
            "description": "List top M crops by production for the given states and years (exact years only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "states": {"type": "array", "items": {"type": "string"}},
                    "years": {"type": "array", "items": {"type": "integer"}},
                    "top_m": {"type": "integer"},
                },
                "required": ["states", "years", "top_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_crop_production",
            "description": "Return total crop production (sum of all crops) for the given states and years (exact years).",
            "parameters": {
                "type": "object",
                "properties": {
                    "states": {"type": "array", "items": {"type": "string"}},
                    "years": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["states", "years"],
            },
        },
    },
    # NEW correlation tool
    {
        "type": "function",
        "function": {
            "name": "get_rainfall_crop_correlation",
            "description": "Compute Pearson correlation between rainfall and total crop production for a state over the last N years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "last_n": {"type": "integer"}
                },
                "required": ["state", "last_n"],
            },
        },
    },
]

available_functions = {
    "get_rainfall_data": get_rainfall_data,
    "get_top_crops": get_top_crops,
    "get_total_crop_production": get_total_crop_production,
    "get_rainfall_crop_correlation": get_rainfall_crop_correlation,
}

# ---------------- Conversation (multi-tool + final “Answer:” block)  ----------------
def run_conversation_groq(user_question: str) -> str:
    print(f"User Question: {user_question}")
    # Guard: if client not initialized, return a helpful error string for the UI
    if client is None:
        return "Conversation error: GROQ API key not configured. Please set GROQ_API_KEY in your .env to enable the chatbot."

    system_rules = (
        "You may call multiple different tools to answer the user's question, "
        "but never call the same function twice. "
        "- rainfall for states/years → get_rainfall_data\n"
        "- top M crops → get_top_crops (with M)\n"
        "- total/overall crop production → get_total_crop_production\n"
        "- if user asks to analyze/relate/impact/correlation → get_rainfall_crop_correlation (choose sensible last_n if not given)\n"
        "If a requested crop year has no data, still answer by showing the nearest available crop year (the tool will note it). "
        "When tools finish, produce a final natural-language answer that directly addresses the question, "
        "including comparisons like 'which is higher' if implied. Keep the source lines from tool outputs. "
        "**Crucially, quote the exact numerical results from the given data in your final answer for precision.**"
    )

    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": user_question},
    ]

    try:
        first = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools, tool_choice="auto",
            max_tokens=2048, temperature=0.1
        )
        msg = first.choices[0].message
        tool_calls = msg.tool_calls
        tool_outputs = []
        messages.append(msg)

        if tool_calls:
            seen = set()
            raw_tool_results = []
            
            for call in tool_calls:
                fn_name = call.function.name
                if fn_name in seen:
                    continue
                seen.add(fn_name)
                args = parse_args(call.function.arguments) or {}

                # light defaults only if the model forgot
                if fn_name == "get_rainfall_data":
                    args.setdefault("states", ["Kerala"])
                    args.setdefault("years", [2010])
                    result = available_functions[fn_name](**args)
                elif fn_name == "get_top_crops":
                    args.setdefault("states", ["Uttar Pradesh"])
                    args.setdefault("years", [2013])
                    args.setdefault("top_m", 3)
                    result = available_functions[fn_name](**args)
                elif fn_name == "get_total_crop_production":
                    args.setdefault("states", ["Uttar Pradesh"])
                    args.setdefault("years", [2013])
                    result = available_functions[fn_name](**args)
                elif fn_name == "get_rainfall_crop_correlation":
                    args.setdefault("state", "Maharashtra")
                    args.setdefault("last_n", 10)
                    result = available_functions[fn_name](**args)
                else:
                    result = f"Unknown function: {fn_name}"

                raw_tool_results.append(result)
                messages.append({
                    "tool_call_id": call.id, "role": "tool", "name": fn_name, "content": result
                })

            # Final NLU pass → concise conclusion from tool outputs
            final_instructions = (
                "Using ONLY given data, answer the user's question directly. "
                "If it implies a comparison, state who is higher and the margin if visible. "
                "If correlation was requested, restate r and what it means in one line. "
                "**Crucially, quote the exact numerical results from the given data in your final answer for precision.**" # Reinforce this
                "Return a short, clear paragraph."
            )
            messages.append({"role": "system", "content": final_instructions})
            messages.append({"role": "user", "content": f"Question: {user_question}Use the given data to answer."})

            final = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_tokens=300, temperature=0.0
            )
            conclusion = final.choices[0].message.content.strip()

            # --- CUSTOM FORMATTING FOR FINAL OUTPUT ---
            
            formatted_data_blocks = []
            sources = set()
            
            for result in raw_tool_results:
                parts = result.split('\n')
                
                # 1. Extract and store sources
                sources.update([p for p in parts if p.startswith('[Source:')])
                
                # 2. Extract clean data (remove sources and the separator line '---')
                clean_lines = [p for p in parts if not p.startswith('[Source:') and p != '---']
                
                if not clean_lines:
                    continue

                # 3. Separate heading from data. The first line is the main heading (e.g., "Rainfall").
                heading = clean_lines[0]
                data_lines = clean_lines[1:]
                
                # 4. Format the data block with the heading and bullets
                # Add a blank line separator for sub-blocks inside Data Used
                if formatted_data_blocks:
                    formatted_data_blocks.append("")
                
                data_block = [
                    f"##### {heading}", # Sub-heading for the specific data
                    "---"
                ]
                # Convert the remaining lines to bullet points
                data_block.extend([f"• {line.strip()}" for line in data_lines if line.strip()])
                formatted_data_blocks.extend(data_block)
            
            # 5. Assemble the final output
            output_parts = []
            
            # --- Citation Block (FIRST HEADING) ---
            output_parts.append("### Citation") 
            output_parts.append("---")
            # Using bullets (•) for sources
            output_parts.extend([f"• {s}" for s in sorted(list(sources))])
            
            output_parts.append("\n") 

            # --- Answer Block (SECOND HEADING) ---
            output_parts.append("# Answer")
            output_parts.append("---")
            output_parts.append(conclusion)
            
            output_parts.append("") 

            # --- Data Used (LAST) ---
            output_parts.append("### Data(traceable)")
            output_parts.append("---")
            output_parts.extend(formatted_data_blocks)

            return "\n".join(output_parts)

        # If the model didn't call any tool (rare), return its text
        return msg.content.strip()

    except Exception as e:
        return f"Conversation error: {e}"

# ---------------- CLI ----------------
if __name__ == "__main__":
    print(f"--- Groq {MODEL_NAME} Chatbot Initialized ---")
    print("Ask about rainfall, top crops, totals, correlation, or both. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            print(f"Bot:\n{run_conversation_groq(user_input)}\n")
        except (EOFError, KeyboardInterrupt):
            print("Exiting.")
            break