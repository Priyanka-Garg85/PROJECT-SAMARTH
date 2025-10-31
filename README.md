# PROJECT-SAMARTH
Empowering India’s Agricultural Intelligence with Data and Climate Insight.

It enables natural language querying over structured data, ensuring **accuracy, traceability, and local data security** — key for agricultural analytics and policy research.

---

## 🚀 Overview

Project Samarth bridges **large language models (LLMs)** and **structured agricultural data**.  
When a user asks a question (e.g., *“Compare rice yield in Maharashtra between 2000 and 2010”*), the system dynamically:

1. **Interprets** the question using Groq’s **Llama-3.1-8B-Instant** model.  
2. **Generates and executes SQL queries** against a local SQLite database (`database.db`).  
3. **Returns traceable, structured results** containing:
   - 📘 **Citation:** Source dataset  
   - 💡 **Answer:** Concise interpretation  
   - 📊 **Data Used:** Raw SQL output rows  

This design guarantees transparency and reproducibility — every answer is grounded in real data.

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

---

## 🧪 Example Query

> **Q:** *"What was the total area under Rice cultivation in Maharashtra in 2010, and how did its yield compare to 2000?"*  
>  
> ✅ **A:** The system dynamically compiles an SQL query, executes it locally, and returns structured, traceable results showing both years’ area and yield side-by-side.

---

## 🔒 Key Highlights

- **Traceable Outputs:** Each answer includes raw SQL data.  
- **Zero Hallucination:** The LLM only interprets — all numbers come directly from the database.  
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

4. **Add your Groq API key:**
   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

Your local chatbot will now be live at **http://localhost:8501** 🎉

---

## 🧭 Future Enhancements

- Integration of additional datasets (soil health, irrigation patterns)  
- Advanced multi-hop querying across datasets  
- Support for Hindi and regional language input  
- Richer visualizations with Altair / Plotly  

---

### 📜 License

This project is released under the **MIT License**.  
Feel free to use, modify, and build upon it with attribution.

---

**Author:** Priyanka Agg  
**Project:** Project Samarth — *Data-Driven Agricultural Q&A System*
