
# 📈 AI Financial Trading Assistant

An interactive AI-powered trading assistant built with Streamlit, CrewAI, and LangChain to analyze stocks, assess risks, generate trading strategies, and suggest execution plans — all in real time using GPT agents.

---

## 🚀 Features

- **Real-Time Stock Analysis** using `yfinance`
- **Multi-Agent Collaboration** via CrewAI (Data Analyst, Risk Specialist, Strategy Developer, Trade Advisor)
- **News & Web Scraping Tools** (Serper API + Web Scraper)
- **Custom Trading Inputs** including capital, risk tolerance, and strategy preference
- **Detailed Agent Reports** with automatic summarization
- **Persistent Chat History** for recent trading analyses

---

## 🧩 Technologies Used

- [Streamlit](https://streamlit.io/)
- [CrewAI](https://docs.crewai.com/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API (GPT-3.5-turbo)](https://platform.openai.com/)
- [Serper.dev](https://serper.dev/) (for search)
- `yfinance` (market data)

---

## 📂 File Structure

```bash
.
├── .env                       # Environment variables (API keys)
├── .gitignore                # Git ignore list
├── app.py                   # Main Streamlit app file
├── main.py                  # (Optional) Auxiliary script
├── requirements.txt         # Python dependencies
├── README_Financial_Analysis.md  # ← This file
└── .venv/                   # Python virtual environment
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-assistant.git
cd ai-trading-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file or use the Streamlit sidebar to input:

```env
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
```

> **Note**: These can also be securely entered via the sidebar UI at runtime.

---

## 🧠 How It Works

1. **User inputs** stock symbol, initial capital, risk tolerance, and strategy type.
2. **CrewAI Agents** analyze the stock from different perspectives:
   - Market trends
   - Risk factors
   - Strategy generation
   - Execution planning
3. A **summary LLM** condenses their findings into a markdown report.
4. **Results** are shown interactively, with collapsible agent outputs and a final summary.
5. **Chat History** allows quick reference to recent analyses.

---

## 📌 Example Use

Input:

- Stock: `AAPL`
- Capital: `$100,000`
- Risk: `Medium`
- Strategy: `Swing Trading`

Output:

- 📊 Market insights from Data Analyst
- ⚠️ Risk profile from Risk Specialist
- 📈 Strategy from Strategy Developer
- 📦 Trade plan from Trade Advisor
- ✅ Final Summary

---

## 📎 To-Do / Improvements

- Add chart visualizations with `plotly` or `matplotlib`
- Enable historical backtesting results
- Support multiple tickers in batch mode
- Add email export of summary

---

## 📃 License

This project is for educational and research use only. Not financial advice.

---

## 🧠 Disclaimer

This tool provides **AI-generated insights** based on public data and models. It is **not a substitute for professional financial advice**. Use at your own risk.
