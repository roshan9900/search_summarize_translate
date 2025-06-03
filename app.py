import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.tools.tavily_search import TavilySearchResults
from sarvamai import SarvamAI
import streamlit as st

# Load API keys
load_dotenv(find_dotenv())
tavily_key = os.getenv("TAVILY_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
sarvam_key = os.getenv("SARVAM_API_KEY")

# Initialize tools
try:
    search = TavilySearchResults(max_results=3, api_key=tavily_key)
except Exception as e:
    st.error(f"Error initializing Tavily search: {e}")
    search = None

try:
    model = init_chat_model(
        model="gemma2-9b-it",  # or mistral-7b
        model_provider="groq",
        api_key=groq_key,
    )
except Exception as e:
    st.error(f"Error initializing Groq model: {e}")
    model = None

try:
    sarvam = SarvamAI(api_subscription_key=sarvam_key)
except Exception as e:
    st.error(f"Error initializing SarvamAI client: {e}")
    sarvam = None

# UI
st.title("🌐 Search → Summarize → Translate")

question = st.text_input("Enter your question")
target_lang = st.selectbox("Translate to language", [
    "bn-IN",  # Bengali
    "gu-IN",  # Gujarati
    "hi-IN",  # Hindi
    "kn-IN",  # Kannada
    "ml-IN",  # Malayalam
    "mr-IN",  # Marathi
    "or-IN",  # Odia
    "pa-IN",  # Punjabi
    "ta-IN",  # Tamil
    "te-IN"   # Telugu
])

btn = st.button("Run")

if btn and question:
    if not search or not model or not sarvam:
        st.error("One or more services failed to initialize. Please check your API keys and setup.")
    else:
        # Step 1: Tavily Search
        try:
            with st.spinner("Fetching context..."):
                result = search.invoke(question)
                context = result[0]["content"] if result and "content" in result[0] else "No context found."
        except Exception as e:
            st.error(f"Error during Tavily search: {e}")
            context = ""

        # Step 2: Groq summarization
        summary = ""
        if context:
            try:
                with st.spinner("Summarizing using Groq..."):
                    prompt = f"""
You are a helpful assistant. Based on the given question and context, provide a short and meaningful summary. Also add an example if possible.

### Question:
{question}

### Context:
{context}

Based on the context above, answer the question in a concise summary.
If the context is not relevant, say "No relevant information found."
"""
                    summary_msg = model.invoke([HumanMessage(content=prompt)])
                    summary = summary_msg.content
            except Exception as e:
                st.error(f"Error during Groq summarization: {e}")

        if summary:
            st.subheader("📝 English Summary")
            st.write(summary)
        else:
            st.warning("No summary available to translate.")

        # Step 3: SarvamAI Translation
        if summary:
            try:
                with st.spinner("Translating using SarvamAI..."):
                    translation = sarvam.text.translate(
                        input=summary,
                        source_language_code="auto",
                        target_language_code=target_lang,
                        speaker_gender="Male"
                    )
                st.subheader("🌍 Translated Summary")
                # Safely access translation result
                translated_text = getattr(translation, "translated_text", None) or translation.get("translated_text", None)
                if translated_text:
                    st.write(translated_text)
                else:
                    st.warning("Translation returned empty result.")
            except Exception as e:
                st.error(f"Error during translation: {e}")
