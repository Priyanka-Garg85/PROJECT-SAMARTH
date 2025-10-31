# PROJECT-SAMARTH
Empowering India’s Agricultural Intelligence with Data and Climate Insight.

It enables natural language querying over structured data, ensuring **accuracy, traceability, and local data security** — key for agricultural analytics and policy research.

---

## 🚀 Overview

Project Samarth with two interactive : 1. Visual Representation 2 . Chatbot
---
1. Visual Representation

This interactive Analytical Insights Dashboard is designed to enable comparative analysis of climate and agricultural trends across India with ease and precision. The visual layout draws inspiration from official Government of India project dashboards, ensuring clarity, accessibility, and high information density.

To begin your exploration, use the following controls:

Select State(s) to Compare:
Choose one or more States or Union Territories to benchmark and compare their climate and agricultural production metrics side-by-side.

Year Range Selection:
Define the temporal range for your analysis.
Note:

Rainfall data is available from 1901–2015

Crop production data spans 2009–2015

Mode of Data:
Select the desired analytical dimension:

Rainfall (Time-Series) – visualize historical rainfall patterns

Crop Totals (Aggregated Production) – analyze total production volumes

Top Crops (Principal Production List) – identify dominant crops per region

Rainfall vs Crop Correlation (Statistical Analysis) – explore the relationship between rainfall and crop yields

Once parameters are set, click Generate to view the corresponding time-series charts, aggregated production metrics, or correlation analyses.

---
2 . Chatbot
## 🧪 Example Query

> **Q:** *"What was the total area under Rice cultivation in Maharashtra in 2010, and how did its yield compare to 2000?"*  
>  
> ✅ **A:** The system dynamically compiles an SQL query, executes it locally, and returns structured, traceable results showing both years’ area and yield side-by-side.
---

## 🧩 System Architecture

```
User Query → LLM (Groq + Function Calling)
           → SQL Query Generation
           → SQLite Database (Local)
           → Structured Response (Citation | Answer | Data)
           → Streamlit Frontend
```

The architecture prioritizes **speed**, **data integrity**, and **privacy** — all processing occurs locally, ensuring zero external data leakage.

---

## 🧠 Tech Stack

| Layer | Technology | Description |
|-------|-------------|-------------|
| 💬 **LLM Engine** | **GroqCloud (Llama-3.1-8B-Instant)** | Fast, low-latency model for natural language → SQL translation |
| 🗄️ **Database** | **SQLite** | Local database for secure, offline data storage and querying |
| 🧮 **Backend** | **Python (chatbot.py)** | Implements RAG logic, SQL function calling, and data canonicalization |
| 💻 **Frontend** | **Streamlit (app.py)** | Interactive chat interface with clean, minimal UI |
| 🧰 **Libraries** | `pandas`, `sqlite3`, `requests`, `streamlit`, `groq` | Data handling, API integration, and UI |

---

## 🌾 Datasets Used

1. **Indian Rainfall Dataset**  
   - Source: Government of India / data.gov.in  
   - Contains annual rainfall data (state-wise) across multiple decades.  
   - Used to correlate climate patterns with crop performance.

2. **State-wise Crop Production Dataset**  
   - Source: Government of India / Ministry of Agriculture  
   - Covers 30+ crops, with data on **area**, **production**, and **yield** by **state** and **year**.  
   - Core dataset powering all agricultural queries.

Both datasets were cleaned, normalized, and merged into a unified **SQLite database (`database.db`)** for seamless querying.

## 🔒 Key Highlights

- **Traceable Outputs:** Each answer includes raw SQL data.  
- **Secure by Design:** No cloud-based data storage or external APIs for sensitive data.  
- **High Speed:** Powered by **Groq’s low-latency inference** for real-time interactivity.  

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/PROJECT-Samarth.git
   cd PROJECT-Samarth
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # (on Mac/Linux)
   venv\Scripts\activate    # (on Windows)
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key with a separate file name .env only for API key :**
   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

Your local chatbot will now be live at localhost

---

## 🧭 Future Enhancements

- Integration of additional datasets (soil health, irrigation patterns)  
- Advanced multi-hop querying across datasets  
- Support for Hindi and regional language input  

---
