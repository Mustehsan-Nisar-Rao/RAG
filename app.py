import streamlit as st
import os
import json
import tempfile
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import requests
import zipfile
import io

# Your hardcoded API key
GEMINI_API_KEY = "AIzaSyCKd3GEjKyvasR4pPktPJVEjRMxIhy7Z2o"

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/Mustehsan-Nisar-Rao/RAG/raw/main/mimic-iv-ext-direct-1.0.zip"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            st.info("📥 Downloading data from GitHub...")
            
            # Use raw GitHub URL
            response = requests.get(self.github_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with open(self.zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = int(50 * downloaded / total_size)
                                progress_bar.progress(min(progress, 100))
                                status_text.text(f"Downloaded {downloaded}/{total_size} bytes")
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Successfully downloaded data from GitHub")
                return True
            else:
                st.error(f"❌ Failed to download file. HTTP Status: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error downloading from GitHub: {e}")
            return False
        
    def extract_data(self):
        """Extract data from ZIP file"""
        # First, download the file if it doesn't exist
        if not os.path.exists(self.zip_path):
            if not self.download_from_github():
                return False
            
        try:
            # Create extraction directory
            os.makedirs(self.extracted_path, exist_ok=True)
            
            # Extract ZIP file
            st.info("📦 Extracting ZIP file...")
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get file list and set up progress
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract all files
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, self.extracted_path)
                    progress = int(100 * (i + 1) / total_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Extracting files... {i+1}/{total_files}")
                
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Successfully extracted data from ZIP file")
            return True
            
        except Exception as e:
            st.error(f"❌ Error extracting ZIP file: {e}")
            return False

# ... (Keep all the other classes exactly the same: SimpleDataProcessor, SimpleRAGSystem, MedicalAI)

def main():
    st.set_page_config(
        page_title="Medical RAG System",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Diagnosis Assistant")
    st.markdown("Ask medical questions about symptoms, diagnoses, and patient cases")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'medical_ai' not in st.session_state:
        st.session_state.medical_ai = None
    if 'data_extracted' not in st.session_state:
        st.session_state.data_extracted = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Show API key status (but don't ask for input)
    st.sidebar.success("🔑 API key configured")
    
    # Data extraction section
    st.sidebar.subheader("📁 Data Setup")
    
    if not st.session_state.data_extracted:
        if st.sidebar.button("📥 Download & Extract Data", type="primary"):
            with st.spinner("Downloading data from GitHub and extracting..."):
                extractor = DataExtractor()
                if extractor.extract_data():
                    st.session_state.data_extracted = True
                    st.session_state.extractor = extractor
                    st.rerun()

    # Initialize system
    if st.session_state.data_extracted and not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            try:
                with st.spinner("🚀 Processing medical data and setting up RAG system... This may take a few minutes."):
                    # Initialize processor and extract data
                    processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                    chunks = processor.run()

                    if not chunks:
                        st.error("❌ No data was extracted. Please check your data file structure.")
                        return

                    # Initialize RAG system
                    rag_system = SimpleRAGSystem(chunks)
                    rag_system.create_collections()
                    rag_system.index_data()

                    # Initialize Medical AI with hardcoded API key
                    st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True

                st.success("✅ System initialized successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")

    # Main interface
    if st.session_state.initialized and st.session_state.medical_ai:
        st.header("💬 Medical Query Interface")

        # Question input
        question = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of migraine? How is chest pain evaluated? What are risk factors for gastrointestinal bleeding?",
            height=100
        )

        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)
            with col2:
                show_context = st.checkbox("Show retrieved context", value=False)

        if st.button("Get Medical Answer", type="primary", use_container_width=True) and question:
            with st.spinner("🔍 Analyzing medical context and generating answer..."):
                try:
                    # Get answer
                    answer = st.session_state.medical_ai.ask(question)

                    # Display answer
                    st.subheader("🤖 Medical Answer")
                    st.markdown(f"**Question:** {question}")
                    st.markdown("**Answer:**")
                    st.write(answer)

                    # Show context if requested
                    if show_context:
                        st.subheader("📚 Retrieved Context")
                        context_chunks = st.session_state.rag_system.query(question, top_k=top_k)
                        
                        for i, chunk in enumerate(context_chunks):
                            with st.expander(f"Context Chunk {i+1}"):
                                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "What are the diagnostic criteria for migraine?",
            "How is chest pain evaluated in emergency settings?",
            "What are common risk factors for gastrointestinal bleeding?",
            "Describe the symptoms and diagnosis process for pneumonia",
            "What are the treatment options for asthma?",
            "How to diagnose and manage diabetes?"
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, use_container_width=True):
                    st.session_state.last_question = example
                    st.rerun()

        # System info
        with st.expander("📊 System Information"):
            if st.session_state.rag_system:
                knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
                narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
                reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
                
                st.write(f"**Knowledge chunks:** {knowledge_count}")
                st.write(f"**Case narratives:** {narrative_count}")
                st.write(f"**Case reasoning:** {reasoning_count}")
                st.write(f"**Total chunks:** {len(st.session_state.rag_system.chunks)}")

    else:
        st.info("""
        👋 **Welcome to the Medical RAG System!**
        
        To get started:
        1. 📥 Click 'Download & Extract Data' to get medical data from GitHub
        2. 🚀 Click 'Initialize System' to build the RAG system
        
        *API key is pre-configured*
        *Data source: https://github.com/Mustehsan-Nisar-Rao/RAG/raw/main/mimic-iv-ext-direct-1.0.zip*
        """)

if __name__ == "__main__":
    main()
