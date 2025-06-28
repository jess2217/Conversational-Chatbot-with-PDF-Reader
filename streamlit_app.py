import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Package Installation Fallback ---
def ensure_packages():
    """Install required packages if missing"""
    try:
        from pypdf import PdfReader
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        from pypdf import PdfReader

ensure_packages()

# --- App Initialization ---
def initialize_app():
    """Set up the app with proper configuration"""
    st.set_page_config(
        page_title="Chatbot with PDF Reader",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Initialize all session variables
    session_defaults = {
        "conversation": None,
        "chat_history": [],
        "processed": False,
        "vectorstore": None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Document Processing ---
def process_pdf_files(uploaded_files):
    """Handle PDF processing pipeline"""
    with st.status("Processing documents...", expanded=True) as status:
        try:
            # Step 1: Extract text
            st.write("📖 Extracting text from PDFs...")
            raw_text = get_pdf_text(uploaded_files)
            if not raw_text:
                status.update(label="No text found in documents", state="error")
                return False
            
            # Step 2: Create chunks
            st.write("✂️ Splitting text into chunks...")
            text_chunks = get_text_chunks(raw_text)
            
            # Step 3: Create vector store
            st.write("🧠 Building knowledge base...")
            vectorstore = get_vectorstore(text_chunks)
            if not vectorstore:
                status.update(label="Failed to create knowledge base", state="error")
                return False
            
            # Step 4: Initialize conversation
            st.write("💡 Setting up AI engine...")
            st.session_state.conversation = setup_conversation(vectorstore)
            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True
            
            status.update(label="✅ Processing complete!", state="complete")
            st.balloons()
            return True
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            status.update(label="❌ Processing failed", state="error")
            return False

# --- Core Functions ---
def get_pdf_text(pdf_docs):
    """Extract text from PDF files with error handling"""
    text = ""
    for pdf in pdf_docs:
        try:
            with st.spinner(f"Reading {pdf.name}..."):
                reader = PdfReader(pdf)
                text += "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"⚠️ Error reading {pdf.name}: {str(e)}")
            continue
    return text if text.strip() else None

def get_text_chunks(text):
    """Split text into chunks for processing"""
    if not text:
        return None
    
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(chunks):
    """Create FAISS vector store from text chunks"""
    if not chunks:
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_texts(chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        return None

def setup_conversation(vectorstore):
    """Initialize the conversation chain with Groq"""
    try:
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            return_source_documents=True
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            verbose=True
        )
    except Exception as e:
        st.error(f"AI setup failed: {str(e)}")
        return None

# --- UI Components ---
def display_chat():
    """Render the conversation history"""
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if i % 2 == 0 else "assistant"):
            st.write(message.content)
            if i % 2 == 1 and hasattr(message, 'source_documents'):
                with st.expander("Source References"):
                    for doc in message.source_documents:
                        st.text(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:200]}...")

def show_processing_sidebar():
    """Render the document processing controls"""
    with st.sidebar:
        st.header("📂 Document Processing")
        st.markdown("""
        **How to use:**
        1. Upload PDF files
        2. Click 'Process Documents'
        3. Ask questions about the content
        """)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze"
        )
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file")
            else:
                if process_pdf_files(uploaded_files):
                    st.success("Documents processed successfully!")
                    st.session_state.processed = True

# --- Main App ---
def main():
    initialize_app()

    st.title("📄 Chatbot with PDF Reader")
    st.caption("Extract knowledge from your documents using AI")
    
    # Document processing sidebar
    show_processing_sidebar()
    
    # Chat interface
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.conversation:
            st.warning("Please process documents first")
        else:
            try:
                with st.spinner("Generating answer..."):
                    response = st.session_state.conversation({'question': prompt})
                    st.session_state.chat_history = response['chat_history']
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display conversation history
    display_chat()

if __name__ == "__main__":
    main()