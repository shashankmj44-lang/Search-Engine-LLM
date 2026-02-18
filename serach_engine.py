import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# ---------------- TOOLS ---------------- #

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun()

tools = [arxiv, wiki, search]

# ---------------- UI ---------------- #

st.title("Shashank's Search Engine 🔎")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.sidebar.warning("Please enter your Groq API key")

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I can search the web, Wikipedia, and Arxiv. What would you like to know?"
        }
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT INPUT ---------------- #

if prompt := st.chat_input("Ask something..."):

    if not api_key:
        st.warning("Please enter your Groq API key before asking a question.")
        st.stop()

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # Initialize LLM
        llm = ChatGroq(
            model="qwen/qwen3-32b",
            groq_api_key=api_key,
            streaming=False
        )

        # Create agent
        agent = create_agent(model=llm, tools=tools)

        # Show spinner while thinking
        with st.spinner("Searching and thinking..."):
            response = agent.invoke({
                "messages": st.session_state["messages"]
            })

        # Extract final answer
        full_response = response["messages"][-1].content

        # Display assistant response
        st.chat_message("assistant").write(full_response)

        # -------- TOOL USAGE DISPLAY -------- #

        
        

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")
