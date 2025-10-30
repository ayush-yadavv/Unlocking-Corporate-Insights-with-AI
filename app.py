import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile

# Google Cloud & Vertex AI
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

# Google Search
from googleapiclient.discovery import build

# Stock Data
from alpha_vantage.timeseries import TimeSeries

# Data Analysis & Sentiment
from textblob import TextBlob

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Financial Analysis RAG Chatbot")

# --- Authentication & Setup ---

@st.cache_resource
def setup_credentials():
    """
    Sets up Google Cloud credentials using Streamlit secrets.
    Returns the path to the credentials file.
    """
    try:
        # Directly read the gcp_service_account section as a dict from st.secrets
        creds_json = dict(st.secrets["gcp_service_account"])
        with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(creds_json).encode())
            creds_path = temp_file.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]
        return creds_path
    except Exception as e:
        st.error(f"Error setting up credentials: {e}")
        st.stop()

@st.cache_resource
def init_vertexai():
    """
    Initializes Vertex AI client.
    """
    try:
        project_id = st.secrets["vertex_project_id"]
        location = st.secrets["vertex_location"]
        vertexai.init(project=project_id, location=location)
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {e}")
        st.stop()

# --- Helper Functions (from Notebook Modules) ---

# Module 2: File Handling
def upload_file_to_gcs(file_bytes, bucket_name, destination_blob_name):
    """Uploads file bytes to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Upload from bytes
    blob.upload_from_string(file_bytes, content_type='application/pdf')
    
    gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
    return gcs_path

def get_pdf_temp_url(bucket_name, object_name):
    """Downloads a PDF from GCS and returns a temporary file path."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    pdf_bytes = blob.download_as_bytes()
    
    with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_bytes)
        temp_file_path = temp_file.name
    
    return temp_file_path

# Module 4: Get Company Details
@st.cache_data
def get_company_details(company_name):
    """Extracts general company details using Gemini."""
    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=[
            "you have a company name as input, generate only the data asked in prompt in this format-",
            "field x: generated data relevant to field x",
            "; field y: generated data relevant to field y"
        ]
    )
    prompt = f"for this company: {company_name}, generate following fields- Company website, Relevant Industries, List of close competitors, Products/services"
    
    response = model.generate_content(
        [prompt],
        generation_config={"max_output_tokens": 8192, "temperature": 0.5, "top_p": 0.95},
    )
    
    # Parse the response
    details = {}
    fields = response.text.split(';')
    for field in fields:
        if ':' in field:
            key, value = field.split(':', 1)
            key = key.strip().lower().replace(' ', '_').replace('/', '_')
            details[key] = value.strip()
    return details

# Module 5: Google Custom Search & Sentiment
@st.cache_data
def get_google_search_results(query):
    """Performs a Google Custom Search."""
    service = build(
        "customsearch", "v1", 
        developerKey=st.secrets["cse_api_key"]
    )
    res = service.cse().list(
        q=query, 
        cx=st.secrets["cse_id"],
        num=5  # Get 5 results
    ).execute()
    return res.get('items', [])

def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob."""
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    if score > 0.1:
        return "Positive", score
    elif score < -0.1:
        return "Negative", score
    else:
        return "Neutral", score

# Module 6: Alpha Vantage Stock Data
@st.cache_data
def get_stock_data(company_name):
    """Finds ticker and gets stock data."""
    av_key = st.secrets["av_api_key"]
    ts = TimeSeries(key=av_key)
    
    # 1. Find Ticker
    try:
        search_data, _ = ts.get_symbol_search(company_name)
        # Prioritize US markets for common tickers
        us_tickers = search_data[search_data['4. region'] == 'United States']
        if not us_tickers.empty:
            ticker = us_tickers.iloc[0]['1. symbol']
        else:
            ticker = search_data.iloc[0]['1. symbol']
    except Exception as e:
        st.warning(f"Could not find ticker symbol: {e}")
        return None, None, None

    # 2. Get Overview
    try:
        overview_data = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={av_key}"
        ).json()
    except Exception:
        overview_data = None

    # 3. Get Intraday Data
    try:
        intraday_data, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
        df = pd.DataFrame.from_dict(intraday_data, orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['4. close'], marker='o', linestyle='-')
        ax.set_title(f'{ticker} Intraday 60-minute Close Prices')
        ax.set_xlabel('Time')
        ax.set_ylabel('Close Price')
        ax.grid(axis='y')
        plt.tight_layout()
        
        return overview_data, df, fig
    
    except Exception as e:
        st.warning(f"Could not retrieve intraday stock data for {ticker}. The API might have limitations. Error: {e}")
        return overview_data, None, None

# Modules 7 & 8: RAG Pipeline
@st.cache_resource
def create_rag_chain(_pdf_url, _company_name, _company_details, _industry_news, _stock_overview):
    """
    Loads PDF, creates vector DB, and returns a runnable RAG chain.
    The arguments starting with _ are to ensure Streamlit's cache
    invalidates when they change.
    """
    with st.spinner("Processing PDF and building knowledge base..."):
        # 1. Load PDF
        pdf_loader = PyPDFLoader(_pdf_url)
        documents = pdf_loader.load()
        
        # 2. Create combined text with context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_text = "\n\n".join([doc.page_content for doc in documents])
        
        # 3. Create Additional Context
        # This context is appended to the document to be embedded
        industry_snippets = [item['snippet'] for item in _industry_news]
        add_context = f"""
        ---
        Additional Context for {company_name}:
        Relevant Industries: {_company_details.get('relevant_industries', 'N/A')}
        Company Offerings: {_company_details.get('products_services', 'N/A')}
        Recent Industry News Snippets: {' | '.join(industry_snippets)}
        Stock Overview: {_stock_overview.get('Description', 'N/A')}
        Market Cap: {_stock_overview.get('MarketCapitalization', 'N/A')}
        52 Week High: {_stock_overview.get('52WeekHigh', 'N/A')}
        52 Week Low: {_stock_overview.get('52WeekLow', 'N/A')}
        ---
        """
        all_text += add_context
        texts = text_splitter.split_text(all_text)
        
        # 4. Create Embeddings
        # We use the same model for embedding and retrieval
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # 5. Create Chroma DB (in-memory)
        # We use a unique collection name to avoid clashes
        collection_name = f"financial_doc_{hash(_pdf_url)}"
        
        # This client is in-memory
        client = chromadb.Client()
        
        # Check if collection exists and delete it (for clean runs)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
            
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_function,
            collection_name=collection_name,
            client=client
        )
        
        # 6. Create RAG Chain
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.7)
        
        # This is the RAG chain that will answer questions
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" is good for smaller contexts
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        
        st.success("Analysis complete. You can now ask questions.")
        return qa_chain

# --- Main Application Logic ---

# Initialize services
try:
    setup_credentials()
    init_vertexai()
    GCS_BUCKET = st.secrets["gcs_bucket_name"]
except Exception as e:
    st.exception(e)
    st.error("Please ensure all secrets are set in your Streamlit configuration.")
    st.stop()


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Configuration")
    company_name_input = st.text_input("Enter Company Name", "Microsoft")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")

    run_button = st.button("Run Analysis")

    # This block will hold the analysis results
    st.header("2. Analysis Summary")
    summary_placeholder = st.empty()

# Initialize session state for chat and chain
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload an annual report and provide a company name, then click 'Run Analysis'."}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Main Processing ---
if run_button:
    if not company_name_input or uploaded_file is None:
        st.error("Please provide both a company name and a PDF file.")
    else:
        # Clear session to force re-analysis
        st.session_state.rag_chain = None
        st.session_state.messages = []
        
        with st.spinner("Starting analysis... This may take a few minutes."):
            # 1. Upload PDF
            pdf_bytes = uploaded_file.getvalue()
            blob_name = f"input_documents/{company_name_input}_annual_report.pdf"
            gcs_path = upload_file_to_gcs(pdf_bytes, GCS_BUCKET, blob_name)
            pdf_temp_url = get_pdf_temp_url(GCS_BUCKET, blob_name)
            
            # 2. Get Company Details (Module 4)
            company_details = get_company_details(company_name_input)
            
            # 3. Get News & Sentiment (Module 5)
            industry_query = f"Industry report and trends for: {company_details.get('relevant_industries', company_name_input)}"
            industry_news = get_google_search_results(industry_query)
            
            company_news = get_google_search_results(f"Latest news for {company_name_input}")

            # Calculate overall sentiment
            all_snippets = " ".join([item['snippet'] for item in industry_news + company_news])
            sentiment, score = analyze_sentiment(all_snippets)

            # 4. Get Stock Data (Module 6)
            stock_overview, stock_df, stock_plot = get_stock_data(company_name_input)
            
            # 5. Create RAG Chain (Modules 7 & 8)
            st.session_state.rag_chain = create_rag_chain(
                pdf_temp_url, 
                company_name_input, 
                company_details, 
                industry_news, 
                stock_overview or {}
            )
            
            # 6. Display Summaries in Sidebar
            with summary_placeholder.container():
                st.subheader("Company Info")
                st.write(f"**Website:** {company_details.get('company_website', 'N/A')}")
                st.write(f"**Industries:** {company_details.get('relevant_industries', 'N/A')}")
                st.write(f"**Competitors:** {company_details.get('list_of_close_competitors', 'N/A')}")
                
                st.subheader("Market Sentiment")
                st.metric(f"Overall News Sentiment", sentiment, f"{score:.2f}")

                if stock_plot:
                    st.subheader("Stock Chart")
                    st.pyplot(stock_plot)
                
                if stock_overview:
                    with st.expander("Stock Overview"):
                        st.json(stock_overview)

            # 7. Update Chat
            st.session_state.messages.append({"role": "assistant", "content": f"Analysis for **{company_name_input}** is complete. You can now ask questions about the annual report and market context."})

# --- Chat Interface (Module 9) ---

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about the financial report..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if analysis has been run
    if st.session_state.rag_chain is None:
        st.warning("Please upload a file and click 'Run Analysis' before asking questions.")
    else:
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the RAG chain
                    response = st.session_state.rag_chain.invoke({"query": prompt})
                    answer = response.get("result", "Sorry, I couldn't find an answer.")
                    
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred: {e}")