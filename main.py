import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="AI Trading Assistant", layout="centered")
st.title("📈 AI Financial Trading Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.header("🔐 API Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")
    submitted_keys = st.button("Ready to Proceed")

    if submitted_keys:
        if not openai_key or not serper_key:
            st.sidebar.warning("Please fill in both API keys.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["SERPER_API_KEY"] = serper_key
            os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
            st.sidebar.success("✅ Now Proceed with an AI Asistant!")

if os.environ.get("OPENAI_API_KEY") and os.environ.get("SERPER_API_KEY"):
    with st.form("trading_form"):
        st.subheader("📋 Trading Configuration")

        stock_selection = st.text_input("Stock Selection", value="AAPL")

        initial_capital = st.text_input("Initial Capital", value="100000")
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        trading_strategy_preference = st.selectbox(
            "Trading Strategy", ["Day Trading", "Swing Trading", "Long-Term Investing"]
        )
        news_impact_consideration = st.checkbox("Consider News Impact", value=True)

        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        try:
            ticker = yf.Ticker(stock_selection)
            stock_info = ticker.info
            real_time_price = stock_info.get("currentPrice") or stock_info.get(
                "regularMarketPrice"
            )
            st.info(f"📊 Real-time Price of {stock_selection}: ${real_time_price}")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch price for {stock_selection}: {e}")

        search_tool = SerperDevTool(search_results_limit=1)
        scrape_tool = ScrapeWebsiteTool()

        data_analyst_agent = Agent(
            role="Data Analyst",
            goal="Monitor and analyze market data in real-time to identify trends.",
            backstory="Specialist in financial modeling using AI tools.",
            verbose=True,
            allow_delegation=True,
            tools=[scrape_tool, search_tool],
        )

        risk_assessment_agent = Agent(
            role="Risk Assessment Specialist",
            goal="Evaluate and assess trading risks based on market conditions and user preferences.",
            backstory="Expert in risk management and portfolio analysis with deep understanding of market volatility.",
            verbose=True,
            allow_delegation=True,
            tools=[scrape_tool, search_tool],
        )

        trading_strategy_agent = Agent(
            role="Trading Strategy Developer",
            goal="Develop profitable trading strategies.",
            backstory="Expert in trading algorithms and backtesting.",
            verbose=True,
            allow_delegation=True,
            tools=[scrape_tool, search_tool],
        )

        execution_agent = Agent(
            role="Trade Advisor",
            goal="Suggest optimal trade execution strategies.",
            backstory="Expert in optimizing order placement for best returns.",
            verbose=True,
            allow_delegation=True,
            tools=[scrape_tool, search_tool],
        )

        data_analysis_task = Task(
            description=f"""Analyze market trends and predict opportunities for {stock_selection}.
Include at least two detailed insights based on technical indicators, price movements, and sentiment.""",
            expected_output=f"Market analysis report with at least 2 lines of insights about {stock_selection}.",
            agent=data_analyst_agent,
        )

        risk_assessment_task = Task(
            description=f"""Assess the risk for trading {stock_selection} based on {risk_tolerance} tolerance.
Include at least two key risk points such as volatility, position sizing, and capital exposure.""",
            expected_output="Risk report with at least two separate risk insights or recommendations.",
            agent=risk_assessment_agent,
        )

        strategy_development_task = Task(
            description=f"""Create a strategy for {stock_selection} using {trading_strategy_preference} based on earlier analysis and risk assessment.
Include at least two clear action steps or strategy components.""",
            expected_output="Trading strategy with at least 2 key recommendations and rationale.",
            agent=trading_strategy_agent,
        )

        execution_planning_task = Task(
            description=f"""Based on the strategy and risk profile for {stock_selection}, suggest how to execute the trade effectively.
Include at least two specific suggestions like order type, timing, or trade size.""",
            expected_output="Execution plan with at least two distinct tactics or tips.",
            agent=execution_agent,
        )

        crew = Crew(
            agents=[
                data_analyst_agent,
                risk_assessment_agent,
                trading_strategy_agent,
                execution_agent,
            ],
            tasks=[
                data_analysis_task,
                risk_assessment_task,
                strategy_development_task,
                execution_planning_task,
            ],
            manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
            process=Process.hierarchical,
            verbose=True,
        )

        with st.spinner("🧠 Running AI Crew..."):
            inputs = {
                "stock_selection": stock_selection,
                "initial_capital": initial_capital,
                "risk_tolerance": risk_tolerance,
                "trading_strategy_preference": trading_strategy_preference,
                "news_impact_consideration": news_impact_consideration,
            }

            crew_output = crew.kickoff(inputs=inputs)

            try:
                individual_outputs = []
                tasks = [
                    data_analysis_task,
                    risk_assessment_task,
                    strategy_development_task,
                    execution_planning_task,
                ]

                for task in tasks:
                    if hasattr(task, "output") and task.output is not None:
                        individual_outputs.append(str(task.output))
                    else:
                        individual_outputs.append("Task output not available")

                if not any(
                    output != "Task output not available"
                    for output in individual_outputs
                ):
                    if hasattr(crew_output, "tasks_output"):
                        individual_outputs = [
                            str(task_output) for task_output in crew_output.tasks_output
                        ]
                    elif hasattr(crew_output, "task_outputs"):
                        individual_outputs = [
                            str(task_output) for task_output in crew_output.task_outputs
                        ]
                    elif hasattr(crew_output, "raw"):
                        raw_output = str(crew_output.raw)
                        individual_outputs = [raw_output] * 4
                    else:
                        raw_output = str(crew_output)
                        individual_outputs = [raw_output] * 4

                agent_titles = [
                    "📊 Data Analyst",
                    "⚠️ Risk Assessment",
                    "📈 Strategy Developer",
                    "📦 Trade Advisor",
                ]

                st.session_state.chat_history.append(
                    {
                        "query": inputs,
                        "agent_responses": dict(zip(agent_titles, individual_outputs)),
                    }
                )
                st.session_state.chat_history = st.session_state.chat_history[-5:]

            except Exception as e:
                st.error(f"Error processing crew output: {e}")
                raw_output = str(crew_output)
                individual_outputs = [raw_output] * 4
                agent_titles = [
                    "📊 Data Analyst",
                    "⚠️ Risk Assessment",
                    "📈 Strategy Developer",
                    "📦 Trade Advisor",
                ]

        st.subheader("🧠 Agent Insights")
        for title, response in zip(agent_titles, individual_outputs):
            with st.expander(title, expanded=False):
                st.markdown(response)

        summary_prompt = ChatPromptTemplate.from_template(
            """You are a summarization expert. Given the following agent outputs, generate a concise final summary in markdown format within 1000 tokens.

### Agent Outputs
{agent_outputs}

### Final Summary:"""
        )

        summarizer = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        summary_input = "\n\n".join(individual_outputs)

        try:
            final_summary = summarizer.invoke(
                summary_prompt.format_messages(agent_outputs=summary_input)
            )
            st.subheader("📌 Final Summary")
            st.markdown(final_summary.content)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            st.subheader("📌 Raw Output")
            st.markdown(str(crew_output))


if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for entry in reversed(st.session_state.chat_history[-5:]):
        user_query = entry["query"]
        st.markdown(
            f"**🧑‍💼 Stock:** `{user_query['stock_selection']}`, Risk: `{user_query['risk_tolerance']}`, Strategy: `{user_query['trading_strategy_preference']}`"
        )
        for role, answer in entry["agent_responses"].items():
            with st.expander(f"📌 {role}"):
                st.markdown(answer)
        st.markdown("---")
