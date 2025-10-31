# PROJECT-SAMARTH
Empowering India’s Agricultural Intelligence with Data and Climate Insight.

It enables natural language querying over structured data, ensuring **accuracy, traceability, and local data security** — key for agricultural analytics and policy research.

---

## 🚀 Overview

Project Samarth with two interactive : 1. Visual Representation 2 . Chatbot
---
1. Visual Representation
This interactive Analytical Insights Dashboard is designed to facilitate comparative analysis of climate and agricultural trends across India for easeness. The visualizations are inspired by official government project dashboards, ensuring clarity, accessibility, and high information density. To begin your exploration, utilize the controls below:

Select State(s) to Compare: Choose one or more States/Union Territories to instantly benchmark and compare their climate and production metrics side-by-side.
Year Range Selection: Specify the temporal scope of your analysis. Note the distinct data availability periods: Rainfall data spans 1901–2015, and Crop data is available for 2009–2015.
Mode of Data: Select the required analytical output: Rainfall (time-series), Crop totals (aggregated production), Top crops (principal production list), or Rainfall vs Crop correlation (statistical analysis).
Click Generate after setting your parameters to view the corresponding time-series charts, production metrics, or statistical findings.
---
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
   git clone https://github.com/your-username/project-samarth.git
   cd project-samarth
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
