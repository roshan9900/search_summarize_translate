````markdown
# 🌐 Web Summarizer & Translator App

This Streamlit app combines powerful AI services to deliver a seamless experience for searching, summarizing, and translating web content — especially focused on Indian languages.

---

## Features

- **Tavily Search:** Fetches relevant context from the web for your query.
- **Groq AI Summarization:** Generates concise, meaningful summaries with examples.
- **SarvamAI Translation:** Translates summaries into 10+ Indian languages.
- Interactive, user-friendly Streamlit interface.
- Robust error handling for smooth operation.

---

## Demo

Try it yourself by entering a question and selecting the target language for translation!

---

## Supported Languages for Translation

| Language  | Code   |
| --------- | ------ |
| Bengali   | bn-IN  |
| Gujarati  | gu-IN  |
| Hindi     | hi-IN  |
| Kannada   | kn-IN  |
| Malayalam | ml-IN  |
| Marathi   | mr-IN  |
| Odia      | or-IN  |
| Punjabi   | pa-IN  |
| Tamil     | ta-IN  |
| Telugu    | te-IN  |

---

## Getting Started

### Prerequisites

- Python 3.8+
- API keys for:
  - [Tavily](https://tavily.com/) (web search)
  - [Groq](https://groq.com/) (LLM summarization)
  - [SarvamAI](https://sarvamai.com/) (translation)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/roshan9900/search_summarize_translate.git
   cd search_summarize_translate
````

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:

   ```
   TAVILY_API_KEY=your_tavily_api_key
   GROQ_API_KEY=your_groq_api_key
   SARVAM_API_KEY=your_sarvamai_api_key
   ```

---

### Running the App

```bash
streamlit run simpleagent.py
```

Open your browser and visit `http://localhost:8501` to use the app.

---

## How It Works

1. **Input:** You enter a question in the app.
2. **Search:** The app uses Tavily to search for the latest and relevant web context.
3. **Summarize:** Groq AI model (`gemma2-9b-it`) generates a concise summary with an example.
4. **Translate:** SarvamAI translates the summary into the selected Indian language.
5. **Output:** The summarized and translated text is displayed.

---

## Code Highlights

* Uses `langchain` and `langchain_core` for seamless AI model integration.
* Robust exception handling to manage API failures gracefully.
* User selects target language from a dropdown menu supporting multiple Indian languages.
* Clear UI using Streamlit for easy interaction.

---

## Future Improvements

* Add voice input and output support.
* Expand language options for translation.
* Add caching to reduce repeated API calls.
* Improve UI/UX design.

---

## Contributing

Contributions and suggestions are welcome! Feel free to open issues or pull requests.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

Built by Roshan Salunke — connect on [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

Happy Summarizing & Translating! 🚀

```

---

```
