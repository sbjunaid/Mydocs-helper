# 🧠 MyDocs Assistant (Offline & Free - LLaMA3)

A powerful **offline knowledge assistant** built using **LangChain**, **Ollama (LLaMA3)**, and **Streamlit**. This assistant reads your `.txt` documents and answers questions using local embeddings + local language models — no internet or API keys needed!

> ✨ Developed by **Sayed Bakhtiar Junaid**

---

## 🚀 Features

- 🔍 Ask questions about your own `.txt` documents (postmortems, logs, infra notes, etc.)
- 🧠 Local embeddings using `all-MiniLM-L6-v2`
- 💬 Powered by **Gemma 3** via Ollama (fully offline)
- 📁 Supports multiple text files (placed in `docs/` folder)
- ⚡ Built with **LangChain** + **Streamlit**
- ❌ No OpenAI / Hugging Face API needed
- ✅ Works 100% locally – no internet required after setup

---

## 📂 Folder Structure

assignment/
│
├── app.py # Streamlit main app
├── docs/ # Place your .txt knowledge base files here
│ ├── github1.txt
│ ├── debugginglog.txt
│ └── ...
├── .gitignore
├── requirements.txt
└── README.md # You're here!


---

## ⚙️ Setup Instructions

### 1. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Mac/Linux
```
### 2.Install Python dependencies
```bash
pip install -r requirements.txt
```
### 3. Install Ollama and Gemma 3 model
(Download Ollama → Install it → Then pull the model):
```bash
ollama run gemma:2b  # Or use llama3, mistral, etc. if you prefer
```
### 4.▶️ Run the App
```bash
streamlit run app.py
```
## 🙋‍♂️ Developer
Made with 💻, 😤, and ☕ by Sayed Bakhtiar Junaid
