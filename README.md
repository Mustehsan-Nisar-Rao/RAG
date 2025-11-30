# 🏥 Medical Diagnosis Assistant

A powerful Retrieval-Augmented Generation (RAG) system for medical diagnosis assistance, built with Streamlit, ChromaDB, and Google Gemini AI.

![Medical RAG System](https://img.shields.io/badge/Medical-AI-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)

## 🌟 Features

- **Medical Knowledge Base**: Extracts knowledge from structured medical data
- **Patient Case Analysis**: Processes real patient cases and diagnostic reasoning
- **RAG-Powered Q&A**: Retrieval-augmented generation for accurate medical answers
- **Multi-source Data**: Handles knowledge graphs and patient case files
- **Real-time Processing**: Instant answers to medical queries

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini Pro
- **Data Processing**: Custom medical data extractor


## 📁 Project Structure
medical-rag-app/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .streamlit/
│ └── config.toml # Streamlit configuration
└── README.md # Project documentation

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-rag-app.git
   cd medical-rag-app
2.  **Install dependencies**
   pip install -r requirements.txt
Data Sources
The system processes medical data from:

Knowledge Graphs: Diagnostic criteria, symptoms, risk factors

Patient Cases: Real clinical narratives and diagnostic reasoning

Medical Conditions: Migraine, pneumonia, asthma, gastrointestinal issues, and more

💡 Example Queries
"What are the diagnostic criteria for migraine?"

"How is chest pain evaluated in emergency settings?"

"What are common risk factors for gastrointestinal bleeding?"

"Describe the symptoms and diagnosis process for pneumonia"

"What are the treatment options for asthma?"

🔧 Configuration
Environment Variables
toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_gemini_api_key_here"
Customization
Modify SimpleDataProcessor to handle different data structures

Adjust top_k parameter in RAG queries

Customize medical prompt templates in MedicalAI class

🎯 How It Works
Data Extraction: Processes JSON files from medical knowledge bases

Chunking: Creates manageable text chunks from medical data

Embedding: Generates vector embeddings using sentence transformers

Indexing: Stores vectors in ChromaDB for efficient retrieval

Query Processing: Finds relevant medical context for each question

Answer Generation: Uses Gemini AI to generate comprehensive answers

📈 Performance
Processes thousands of medical knowledge chunks

Supports real-time medical Q&A

Handles complex medical terminology

Provides context-aware responses

🤝 Contributing
We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

Development Setup
Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request
   
