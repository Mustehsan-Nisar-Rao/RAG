import streamlit as st
import os
import json
import tempfile
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import patoolib  # For RAR extraction

class DataExtractor:
    def __init__(self):
        self.rar_path = "./data.rar"
        self.extracted_path = "./data_extracted"
        
    def extract_data(self):
        """Extract data from RAR file"""
        if not os.path.exists(self.rar_path):
            st.error(f"❌ RAR file not found: {self.rar_path}")
            return False
            
        try:
            # Create extraction directory
            os.makedirs(self.extracted_path, exist_ok=True)
            
            # Extract RAR file
            patoolib.extract_archive(self.rar_path, outdir=self.extracted_path)
            st.success("✅ Successfully extracted data from RAR file")
            return True
            
        except Exception as e:
            st.error(f"❌ Error extracting RAR file: {e}")
            st.info("💡 Make sure patool is installed and unrar is available")
            return False

class SimpleDataProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        # Try different possible paths after extraction
        self.possible_kg_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "Diagnosis_flowchart"),
        ]
        self.possible_case_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "Finished"),
            os.path.join(base_path, "Finished"),
            os.path.join(base_path, "cases"),
        ]
        
        self.kg_path = self._find_valid_path(self.possible_kg_paths)
        self.cases_path = self._find_valid_path(self.possible_case_paths)
    
    def _find_valid_path(self, possible_paths):
        """Find the first valid path that exists"""
        for path in possible_paths:
            if os.path.exists(path):
                st.info(f"📁 Found path: {path}")
                return path
        return None

    def check_data_exists(self):
        """Check if data directories exist and have files"""
        kg_exists = self.kg_path and os.path.exists(self.kg_path) and any(f.endswith('.json') for f in os.listdir(self.kg_path))
        cases_exists = self.cases_path and os.path.exists(self.cases_path) and any(os.path.isdir(os.path.join(self.cases_path, d)) for d in os.listdir(self.cases_path))
        
        return kg_exists, cases_exists

    def count_files(self):
        """Count all JSON files"""
        kg_count = 0
        if self.kg_path and os.path.exists(self.kg_path):
            kg_count = len([f for f in os.listdir(self.kg_path) if f.endswith('.json')])

        case_count = 0
        if self.cases_path and os.path.exists(self.cases_path):
            for item in os.listdir(self.cases_path):
                item_path = os.path.join(self.cases_path, item)
                if os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        case_count += len([f for f in files if f.endswith('.json')])
                elif item.endswith('.json'):
                    case_count += 1

        st.info(f"📊 Found {kg_count} knowledge files and {case_count} case files")
        return kg_count, case_count

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []

        if not self.kg_path or not os.path.exists(self.kg_path):
            st.error(f"❌ Knowledge graph path not found")
            st.info(f"💡 Checked paths: {self.possible_kg_paths}")
            return chunks

        for filename in os.listdir(self.kg_path):
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(self.kg_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                condition = filename.replace('.json', '')
                knowledge = data.get('knowledge', {})

                for stage_name, stage_data in knowledge.items():
                    if isinstance(stage_data, dict):
                        # Extract risk factors
                        if stage_data.get('Risk Factors'):
                            chunks.append({
                                'text': f"{condition} - Risk Factors: {stage_data['Risk Factors']}",
                                'metadata': {'type': 'knowledge', 'category': 'risk_factors', 'condition': condition}
                            })

                        # Extract symptoms
                        if stage_data.get('Symptoms'):
                            chunks.append({
                                'text': f"{condition} - Symptoms: {stage_data['Symptoms']}",
                                'metadata': {'type': 'knowledge', 'category': 'symptoms', 'condition': condition}
                            })
            except Exception as e:
                st.warning(f"⚠️ Error processing {filename}: {e}")
                continue

        st.success(f"✅ Extracted {len(chunks)} knowledge chunks")
        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        if not self.cases_path or not os.path.exists(self.cases_path):
            st.error(f"❌ Cases path not found")
            st.info(f"💡 Checked paths: {self.possible_case_paths}")
            return chunks

        # Handle both directory structure and flat files
        items = os.listdir(self.cases_path)
        
        for item in items:
            item_path = os.path.join(self.cases_path, item)
            
            if os.path.isdir(item_path):
                # It's a directory (like Migraine/, Pneumonia/, etc.)
                condition_folder = item
                for filename in os.listdir(item_path):
                    if filename.endswith('.json'):
                        self._process_case_file(os.path.join(item_path, filename), condition_folder, chunks)
            elif item.endswith('.json'):
                # It's a JSON file in the root
                condition_folder = "General"
                self._process_case_file(item_path, condition_folder, chunks)

        narratives = len([c for c in chunks if c['metadata']['type'] == 'narrative'])
        reasoning = len([c for c in chunks if c['metadata']['type'] == 'reasoning'])
        st.success(f"✅ Extracted {narratives} narrative chunks and {reasoning} reasoning chunks")
        return chunks

    def _process_case_file(self, file_path, condition_folder, chunks):
        """Process individual case file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            case_id = filename.replace('.json', '')

            # Extract narrative (inputs)
            narrative_parts = []
            for i in range(1, 7):
                key = f'input{i}'
                if key in data and data[key]:
                    narrative_parts.append(f"{key}: {data[key]}")

            if narrative_parts:
                chunks.append({
                    'text': f"Case {case_id} - {condition_folder}\nNarrative:\n" + "\n".join(narrative_parts),
                    'metadata': {'type': 'narrative', 'case_id': case_id, 'condition': condition_folder}
                })

            # Extract reasoning
            for key in data:
                if not key.startswith('input'):
                    reasoning = self._extract_reasoning(data[key])
                    if reasoning:
                        chunks.append({
                            'text': f"Case {case_id} - {condition_folder}\nReasoning:\n{reasoning}",
                            'metadata': {'type': 'reasoning', 'case_id': case_id, 'condition': condition_folder}
                        })
        except Exception as e:
            st.warning(f"⚠️ Error processing {file_path}: {e}")

    def _extract_reasoning(self, data):
        """Simple reasoning extraction"""
        reasoning_lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if '$Cause_' in key:
                    reasoning_text = key.split('$Cause_')[0].strip()
                    if reasoning_text:
                        reasoning_lines.append(reasoning_text)

                if isinstance(value, (dict, list)):
                    nested_reasoning = self._extract_reasoning(value)
                    if nested_reasoning:
                        reasoning_lines.append(nested_reasoning)

        elif isinstance(data, list):
            for item in data:
                nested_reasoning = self._extract_reasoning(item)
                if nested_reasoning:
                    reasoning_lines.append(nested_reasoning)

        return "\n".join(reasoning_lines) if reasoning_lines else ""

    def run(self):
        """Run complete extraction"""
        st.info("🚀 Starting data extraction...")

        # Check if data exists
        kg_exists, cases_exists = self.check_data_exists()
        if not kg_exists and not cases_exists:
            st.error("❌ No valid data found after extraction.")
            return []

        # Count files
        kg_count, case_count = self.count_files()

        if kg_count == 0 and case_count == 0:
            st.error("❌ No JSON files found in data directories.")
            return []

        # Extract data
        knowledge_chunks = self.extract_knowledge()
        case_chunks = self.extract_patient_cases()

        all_chunks = knowledge_chunks + case_chunks

        if all_chunks:
            st.success(f"🎯 Extraction complete: {len(knowledge_chunks)} knowledge + {len(case_chunks)} cases = {len(all_chunks)} total chunks")
        else:
            st.error("❌ No data chunks were extracted")

        return all_chunks

class SimpleRAGSystem:
    def __init__(self, chunks, db_path="./chroma_db"):
        self.chunks = chunks
        self.db_path = db_path
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")

    def create_collections(self):
        """Create separate collections for knowledge and cases"""
        try:
            # Knowledge collection
            self.knowledge_collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                embedding_function=self.embedding_function
            )

            # Cases collection
            self.cases_collection = self.client.get_or_create_collection(
                name="patient_cases",
                embedding_function=self.embedding_function
            )

            st.success("✅ Created ChromaDB collections")
        except Exception as e:
            st.error(f"Error creating collections: {e}")

    def index_data(self):
        """Index all chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        try:
            for i, chunk in enumerate(self.chunks):
                if chunk['metadata']['type'] == 'knowledge':
                    knowledge_docs.append(chunk['text'])
                    knowledge_metas.append(chunk['metadata'])
                    knowledge_ids.append(f"kg_{i}")
                else:
                    case_docs.append(chunk['text'])
                    case_metas.append(chunk['metadata'])
                    case_ids.append(f"case_{i}")

            # Add to collections
            if knowledge_docs:
                self.knowledge_collection.add(
                    documents=knowledge_docs,
                    metadatas=knowledge_metas,
                    ids=knowledge_ids
                )

            if case_docs:
                self.cases_collection.add(
                    documents=case_docs,
                    metadatas=case_metas,
                    ids=case_ids
                )

            st.success(f"✅ Indexed {len(knowledge_docs)} knowledge chunks and {len(case_docs)} case chunks")
        except Exception as e:
            st.error(f"Error indexing data: {e}")

    def query(self, question, top_k=5):
        """Simple query across both collections"""
        try:
            # Query knowledge
            knowledge_results = self.knowledge_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Query cases
            case_results = self.cases_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Combine results
            all_results = []
            if knowledge_results['documents']:
                all_results.extend(knowledge_results['documents'][0])
            if case_results['documents']:
                all_results.extend(case_results['documents'][0])

            return all_results
        except Exception as e:
            st.error(f"Error querying RAG system: {e}")
            return []

class MedicalAI:
    def __init__(self, rag_system, api_key):
        self.rag = rag_system
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")

    def ask(self, question):
        try:
            # Get relevant context from RAG
            context_chunks = self.rag.query(question, top_k=5)
            context = "\n---\n".join(context_chunks)

            # Create prompt
            prompt = f"""You are a medical expert. Use the following medical context to answer the question accurately.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive medical answer based on the context. If the context doesn't contain enough information, state what's missing."""

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

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
    
    # API Key input
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("🔑 API key loaded from secrets")
    else:
        api_key = st.sidebar.text_input("Gemini API Key", type="password")
        if not api_key:
            st.sidebar.warning("Please enter your Gemini API key")

    # Data extraction section
    st.sidebar.subheader("📁 Data Setup")
    
    if not st.session_state.data_extracted:
        if st.sidebar.button("📦 Extract Data from RAR", type="primary"):
            with st.spinner("Extracting data from RAR file..."):
                extractor = DataExtractor()
                if extractor.extract_data():
                    st.session_state.data_extracted = True
                    st.session_state.extractor = extractor
                    st.rerun()

    # Initialize system
    if st.session_state.data_extracted and not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            if not api_key:
                st.error("❌ Please enter your Gemini API key")
                return
                
            try:
                with st.spinner("🚀 Processing medical data and setting up RAG system... This may take a few minutes."):
                    # Initialize processor and extract data
                    processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                    chunks = processor.run()

                    if not chunks:
                        st.error("❌ No data was extracted. Please check your RAR file structure.")
                        return

                    # Initialize RAG system
                    rag_system = SimpleRAGSystem(chunks)
                    rag_system.create_collections()
                    rag_system.index_data()

                    # Initialize Medical AI
                    st.session_state.medical_ai = MedicalAI(rag_system, api_key)
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
                    # Set the question in the text area
                    st.session_state.last_question = example
                    st.rerun()

        # System info
        with st.expander("📊 System Information"):
            if st.session_state.rag_system:
                st.write(f"**Knowledge chunks:** {len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])}")
                st.write(f"**Case narratives:** {len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])}")
                st.write(f"**Case reasoning:** {len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])}")
                st.write(f"**Total chunks:** {len(st.session_state.rag_system.chunks)}")

    else:
        st.info("""
        👋 **Welcome to the Medical RAG System!**
        
        To get started:
        1. 🔑 Enter your Gemini API key in the sidebar
        2. 📦 Click 'Extract Data from RAR' to unpack the medical data
        3. 🚀 Click 'Initialize System' to build the RAG system
        
        *Note: The RAR file (data.rar) must be in the root directory.*
        """)

        # Quick setup guide
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **1. Get Gemini API Key:**
            - Visit [Google AI Studio](https://aistudio.google.com/)
            - Create an API key for Gemini
            
            **2. Prepare Your Data:**
            - Ensure your RAR file is named `data.rar`
            - Place it in the same directory as this app
            - The RAR should contain medical JSON files in the expected structure
            
            **3. Initialize:**
            - Click the buttons in order: Extract → Initialize
            - Wait for the processing to complete
            """)

        # File structure info
        with st.expander("📁 Expected Data Structure in RAR"):
            st.markdown("""
            Your `data.rar` should contain:
            ```
            mimic-iv-ext-direct-1.0/
            └── mimic-iv-ext-direct-1.0.0/
                ├── diagnostic_kg/
                │   └── Diagnosis_flowchart/
                │       ├── migraine.json
                │       ├── pneumonia.json
                │       └── ...
                └── Finished/
                    ├── Migraine/
                    │   ├── case1.json
                    │   └── ...
                    ├── Pneumonia/
                    │   ├── case1.json
                    │   └── ...
                    └── ...
            ```
            """)

if __name__ == "__main__":
    main()
