# 📄 Resumé Sentiment Analysis

This is a Streamlit application designed to evaluate the tone and sentiment of professional résumés. By leveraging Hugging Face Transformers and LangChain, the app breaks down long-form text into manageable chunks and uses Natural Language Processing (NLP) models to determine if the writing is positive and what level of professional seniority it conveys.

![](img/Resume-app.png)

## ✨ Features

* **Intelligent Text Chunking**: Uses RecursiveCharacterTextSplitter from LangChain to handle long résumés while maintaining contextual integrity through token overlaps.
* **Sentiment Analysis**: Detects the overall emotional tone of the text (e.g., Positive/Negative) using a pre-trained sentiment pipeline.
* **Zero-Shot Classification**: Categorizes the professional "tone" or career level of the résumé into labels like Senior, Junior, White-collar, etc., without requiring specific fine-tuning.
* **User-Friendly Interface**: A clean, interactive web dashboard built with Streamlit.

## 🛠️ Technical Stack

|ComponentTool | Library |
|--------------|---------|
|Frontend | Streamlit |
|NLP Framework | Hugging Face Transformers |
|Model (Sentiment) | Default DistilBERT (via `pipeline`) |
|Model (Tone) | `facebook/bart-large-mnli` |
|Text Processing | LangChain (`RecursiveCharacterTextSplitter`) |
|Tokenizer | `bert-base-uncased` |

## 🚀 How It Works

* **Input**: The user pastes résumé text into the provided text area.
* **Tokenization**: The text is processed using the bert-base-uncased tokenizer to ensure the input is compatible with transformer models.
* **Chunking**: Because transformer models have a maximum token limit, the code splits the text into chunks of 500 words with a 100-word overlap to ensure no context is lost at the boundaries.
* **Analysis**:The Sentiment Pipeline evaluates the emotional positivity of the chunks.The Zero-Shot Classification Pipeline compares the text against a list of candidate labels (Senior, Junior, Trainee, etc.).
* **Output**: The app displays the detected sentiment, the professional category, and the confidence scores for both.

## ⚙️ Installation & Usage

1. Clone the repository

```Bash
git clone https://github.com/your-username/resume-sentiment-analysis.git
cd resume-sentiment-analysis
```

2. Install dependencies

Ensure you have Python 3.8+ installed. You will need torch, transformers, langchain, and streamlit.

```Bash
pip install streamlit torch transformers langchain
```

3. Run the App

```Bash
streamlit run app.py
```

## 📝 Code Structure Notes

* `tokenizer.prepare_for_model`: This step is crucial as it automatically adds special tokens like `[CLS]` and `[SEP]` back to the chunks after splitting, ensuring the BERT-based models can process them correctly.
* Multi-Column Layout: The results are displayed side-by-side using st.columns(2) for a better user experience."# Resum-Sentiment-Analysis" 
