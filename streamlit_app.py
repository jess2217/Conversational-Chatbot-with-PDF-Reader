import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM
def get_groq_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",  # Updated to current model name
        api_key=os.getenv("GROQ_API_KEY")
    )

# Function to extract text from PDFs with error handling
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text() or ""  # Handle None
                text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
            continue
    return text if text.strip() else None

# Function to split text into chunks with validation
def get_text_chunks(text):
    if not text:
        return None
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks if chunks else None

# Function to create vector store with error handling
def get_vectorstore(text_chunks):
    if not text_chunks:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Vector store creation failed: {str(e)}")
        return None

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None
    try:
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=get_groq_llm(),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    except Exception as e:
        st.error(f"Failed to create conversation chain: {str(e)}")
        return None

# Function to handle user input with error handling
def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("Conversation not initialized. Please process documents first.")
        return
    
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

# Main Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚")
    st.header("📚 Chat with PDF Chatbot")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User question input
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'", 
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            else:
                with st.spinner("Processing..."):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No readable text found in documents")
                        return
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Failed to create text chunks")
                        return
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    if not vectorstore:
                        st.error("Failed to create vector store")
                        return
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.success(f"Processed {len(pdf_docs)} PDF(s) with {len(text_chunks)} chunks")
                        st.info("You can now ask questions about your documents!")
                    else:
                        st.error("Failed to initialize conversation chain")

if __name__ == '__main__':
    main()
