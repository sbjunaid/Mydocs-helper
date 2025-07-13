import os
import streamlit as st
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # ✅ Ollama auto-runs locally

# ✅ Title
st.title("🧠 Mydoc Assistant (Offline with Gemma3 via Ollama)")

# ✅ Load .txt files
def load_docs():
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith(".txt"):
            path = os.path.join("docs", filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    docs.append(Document(page_content=content, metadata={"source": filename}))
                    st.write(f"✅ Loaded: {filename}")
            except Exception as e:
                st.error(f"❌ Error reading {filename}: {e}")
    return docs

# ✅ Get user query
query = st.text_input("Ask something about GitHub, Python, debugging, etc:")

if query:
    with st.spinner("🔄 Processing..."):
        try:
            # Load documents
            documents = load_docs()
            st.write(f"📄 Total documents loaded: {len(documents)}")
            if not documents:
                st.warning("⚠️ No documents found in 'docs' folder.")
                st.stop()

            # Split into chunks
            st.write("📎 Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
            )
            texts = splitter.split_documents(documents)
            st.write(f"📎 Total chunks created: {len(texts)}")

            # Embeddings
            st.write("🧠 Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.write("✅ Embeddings ready.")

            # Vector store
            st.write("🧠 Creating vector store...")
            vectorstore = Chroma.from_documents(texts, embeddings)
            st.write("✅ Vector store created.")

            # 🧠 Ollama (auto-runs Gemma3 in background)
            st.write("🤖 Loading local Gemma3 model via Ollama...")
            llm = Ollama(model="gemma3", temperature=0.3)
            st.write("✅ Model loaded.")

            # Build QA
            st.write("🔗 Building QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_type="similarity", k=3),
                return_source_documents=True
            )
            st.write("✅ QA chain ready.")

            # Run Query
            with st.spinner("🤖 Thinking..."):
                st.write("💬 Running query...")
                response = qa_chain(query)
                st.success(response["result"])

                with st.expander("🔍 Sources used"):
                    for doc in response["source_documents"]:
                        st.markdown(f"📄 **{doc.metadata['source']}**")
                        st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

        except Exception as e:
            st.error(f"❌ Error in pipeline: {e}")
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Made with 💻, 😤, and ☕ by <strong>Sayed Bakhtiar Junaid</strong> 🙋‍♂️.
    </div>
    """,
    unsafe_allow_html=True
)
