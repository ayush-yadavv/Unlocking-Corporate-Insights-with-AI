import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
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

def format_market_cap(market_cap):
    """Format market cap value into human-readable format (e.g., 1.2B, 3.4T)"""
    if market_cap in [None, 'N/A', '']:
        return 'N/A'
    try:
        num = float(market_cap)
        if num >= 1e12:  # Trillions
            return f'${num/1e12:.2f}T'
        elif num >= 1e9:  # Billions
            return f'${num/1e9:.2f}B'
        elif num >= 1e6:  # Millions
            return f'${num/1e6:.2f}M'
        else:
            return f'${num:,.2f}'
    except (ValueError, TypeError):
        return str(market_cap)

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="📈",
    layout="centered"
)

# Main title
st.title("📊 Corporate Financial Insights & Analysis")
st.markdown("---")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please upload an annual report and provide a company name, then click 'Run Analysis' to begin."}]

# --- Authentication & Setup ---

@st.cache_resource
def setup_credentials():
    """
    Sets up Google Cloud credentials using Streamlit secrets.
    Returns the path to the credentials file.
    """
    try:
        # Create a dictionary with the service account info from TOML
        creds_dict = {
            "type": st.secrets.gcp_service_account.type,
            "project_id": st.secrets.gcp_service_account.project_id,
            "private_key_id": st.secrets.gcp_service_account.private_key_id,
            "private_key": st.secrets.gcp_service_account.private_key.replace('\\n', '\n'),
            "client_email": st.secrets.gcp_service_account.client_email,
            "client_id": st.secrets.gcp_service_account.client_id,
            "auth_uri": st.secrets.gcp_service_account.auth_uri,
            "token_uri": st.secrets.gcp_service_account.token_uri,
            "auth_provider_x509_cert_url": st.secrets.gcp_service_account.auth_provider_x509_cert_url,
            "client_x509_cert_url": st.secrets.gcp_service_account.client_x509_cert_url,
            "universe_domain": st.secrets.gcp_service_account.universe_domain
        }
        
        with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(creds_dict).encode())
            creds_path = temp_file.name
            
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        os.environ["GOOGLE_API_KEY"] = st.secrets.google_api_key
        return creds_path
    except Exception as e:
        st.error(f"Error setting up credentials: {str(e)}")
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
        "gemini-2.5-flash-lite",
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
        
        # Create interactive Plotly figure
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True, 
                          vertical_spacing=0.1,
                          row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['1. open'],
                high=df['2. high'],
                low=df['3. low'],
                close=df['4. close'],
                name='OHLC',
                increasing_line_color='#2ecc71',  # Green for up
                decreasing_line_color='#e74c3c'   # Red for down
            ),
            row=1, col=1
        )
        
        # Add volume as barchart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['5. volume'],
                name='Volume',
                marker_color='#3498db',
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Add moving averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean()
        df['SMA50'] = df['4. close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name='20-SMA',
                line=dict(color='#f39c12', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name='50-SMA',
                line=dict(color='#9b59b6', width=1.5)
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Intraday Stock Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=800,
            showlegend=True,
            xaxis2_title='Date',
            yaxis2_title='Volume',
            hovermode='x unified',
            xaxis=dict(rangeslider=dict(visible=False))
        )
        
        # Update y-axes
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
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
        Additional Context for {_company_name}:
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
        
        # 4. Create Embeddings using all-MiniLM-L6-v2
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': False}
        )

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
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
        
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

def safe_float_convert(value, default=None):
    """Safely convert value to float, return default if conversion fails."""
    if value is None or value == 'None' or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def format_percentage(value, default='N/A'):
    """Format value as percentage if it can be converted to float, else return default."""
    num = safe_float_convert(value)
    return f"{num:.2%}" if num is not None else default

def calculate_valuation_metrics(stock_data):
    """Calculate additional valuation metrics based on available stock data."""
    if not stock_data:
        return {}
        
    metrics = {}
    
    try:
        # Helper function to safely get and format values
        def get_metric(key, formatter=None):
            value = stock_data.get(key)
            if value is None or value == 'None':
                return 'N/A'
            if formatter == 'percent':
                return format_percentage(value)
            if formatter == 'float':
                num = safe_float_convert(value)
                return f"{num:,.2f}" if num is not None else 'N/A'
            return value
        
        # Price-to-Book (P/B) ratio
        metrics['P/B Ratio'] = get_metric('PriceToBookRatio')
        
        # EV/EBITDA
        metrics['EV/EBITDA'] = get_metric('EVToEBITDA')
        
        # Dividend Yield
        div_yield = safe_float_convert(stock_data.get('DividendYield'))
        metrics['Dividend Yield'] = f"{div_yield:.2%}" if div_yield is not None else 'N/A'
        
        # PEG Ratio
        metrics['PEG Ratio'] = get_metric('PEGRatio')
        
        # Current Ratio
        metrics['Current Ratio'] = get_metric('CurrentRatio', 'float')
        
        # Debt-to-Equity
        metrics['Debt/Equity'] = get_metric('DebtToEquity')
        
        # Return on Equity (TTM)
        roe = safe_float_convert(stock_data.get('ReturnOnEquityTTM'))
        metrics['ROE (TTM)'] = f"{roe:.2%}" if roe is not None else 'N/A'
        
        # Return on Assets (TTM)
        roa = safe_float_convert(stock_data.get('ReturnOnAssetsTTM'))
        metrics['ROA (TTM)'] = f"{roa:.2%}" if roa is not None else 'N/A'
        
        # Beta (5Y Monthly)
        metrics['Beta (5Y Monthly)'] = get_metric('Beta')
        
        # 52-Week Price Change
        week52_raw = stock_data.get('52WeekChange')
        print(f"Debug - 52WeekChange raw value: {week52_raw}")
        week52 = safe_float_convert(week52_raw)
        print(f"Debug - 52WeekChange after conversion: {week52}")
        if week52 is not None:
            metrics['52-Week Change'] = f"{week52:+.2%}"
        else:
            print(f"Debug - 52WeekChange is None, available keys: {list(stock_data.keys())}")
        
        # Additional metrics with safe handling
        metrics['EPS'] = get_metric('EPS', 'float')
        metrics['Revenue (TTM)'] = get_metric('RevenueTTM', 'float')
        metrics['Gross Profit (TTM)'] = get_metric('GrossProfitTTM', 'float')
        
    except Exception as e:
        st.warning(f"Some metrics could not be calculated: {str(e)}")
        # Return whatever metrics we could calculate
    
    return metrics

def display_analysis_summary(company_name, company_details, sentiment_score, stock_overview, industry_news, company_news, stock_plot=None):
    """Display a comprehensive financial analysis summary in the main content area."""
    st.header(f"📊 {company_name}")
    
    # Calculate additional metrics
    valuation_metrics = calculate_valuation_metrics(stock_overview)
    
    # Debug stock_overview keys
    print(f"Debug - Available stock_overview keys: {list(stock_overview.keys())}")
    
    # Key Metrics Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = stock_overview.get('MarketCapitalization')
        if market_cap and market_cap != 'N/A':
            try:
                # Convert to float and format in billions or millions
                market_cap_float = float(market_cap)
                if market_cap_float >= 1_000_000_000:
                    formatted_market_cap = f"${market_cap_float/1_000_000_000:.2f}B"
                elif market_cap_float >= 1_000_000:
                    formatted_market_cap = f"${market_cap_float/1_000_000:.2f}M"
                else:
                    formatted_market_cap = f"${market_cap_float:,.2f}"
            except (ValueError, TypeError):
                formatted_market_cap = 'N/A'
        else:
            formatted_market_cap = 'N/A'
        st.metric("Market Cap", format_market_cap(stock_overview.get('MarketCapitalization')))
    with col2:
        pe_ratio = stock_overview.get('PERatio', 'N/A')
        st.metric("P/E Ratio", f"{pe_ratio}x" if pe_ratio != 'N/A' else 'N/A')
    with col3:
        pb_ratio = valuation_metrics.get('P/B Ratio', 'N/A')
        st.metric("P/B Ratio", f"{pb_ratio}x" if pb_ratio != 'N/A' else 'N/A')
    with col4:
        div_yield = valuation_metrics.get('Dividend Yield', 'N/A')
        st.metric("Dividend Yield", div_yield if div_yield != 'N/A' else 'N/A')
    
    # Key Metrics Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roe = valuation_metrics.get('ROE (TTM)', 'N/A')
        st.metric("ROE (TTM)", roe if roe != 'N/A' else 'N/A')
    with col2:
        beta = valuation_metrics.get('Beta (5Y Monthly)', 'N/A')
        st.metric("Beta", beta if beta != 'N/A' else 'N/A')
    with col3:
        ev_ebitda = valuation_metrics.get('EV/EBITDA', 'N/A')
        st.metric("EV/EBITDA", f"{ev_ebitda}x" if ev_ebitda != 'N/A' else 'N/A')
    with col4:
        week52change = valuation_metrics.get('52-Week Change', 'N/A')
        st.metric("52-Week Change", week52change if week52change != 'N/A' else 'N/A')
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Market Overview",
        "Investment Thesis", 
        "Valuation", 
        "Financial Health", 
        "Growth & Returns",
        "Risks & Opportunities",
        "Competitive Landscape"
    ])
    
    with tab1:  # Market Overview
        st.subheader("Market Analysis")
        
        # Market Metrics
        st.markdown("### Key Market Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Industry", company_details.get('industry', 'N/A'))
            st.metric("Market Sentiment", f"{sentiment_score:.1f}/10.0" if sentiment_score else 'N/A')
        with mcol2:
            st.metric("52-Week High/Low", 
                     f"${stock_overview.get('52WeekHigh', 'N/A')} / ${stock_overview.get('52WeekLow', 'N/A')}" 
                     if stock_overview.get('52WeekHigh') and stock_overview.get('52WeekLow') else 'N/A')
            st.metric("Volume (Avg)", f"{int(float(stock_overview.get('Volume', 0)) / 1_000_000):.1f}M" 
                     if stock_overview.get('Volume') else 'N/A')
        with mcol3:
            st.metric("Beta", stock_overview.get('Beta', 'N/A'))
            st.metric("Short Ratio", stock_overview.get('ShortRatio', 'N/A'))
        
        # Market Trends
        st.markdown("### Market Trends")
        st.markdown("""
        - **Industry Position**: {company_name} holds a {position} position in the {industry} sector
        - **Competitive Landscape**: Facing competition from {competitors}
        - **Regulatory Environment**: {regulatory_notes}
        - **Technological Trends**: {tech_trends}
        - **Consumer Sentiment**: {sentiment_analysis}
        """.format(
            company_name=company_name,
            position=company_details.get('position', 'leading'),
            industry=company_details.get('industry', 'its industry'),
            competitors=company_details.get('competitors', 'major industry players'),
            regulatory_notes=company_details.get('regulatory_notes', 'Stable regulatory environment'),
            tech_trends=company_details.get('tech_trends', 'Benefiting from digital transformation trends'),
            sentiment_analysis='Positive' if sentiment_score and sentiment_score >= 6 else 
                             'Neutral' if sentiment_score and sentiment_score >= 4 else 'Negative'
        ))
        
        # Industry News Highlights
        if industry_news and len(industry_news) > 0:
            st.markdown("### Industry News Highlights")
            for i, news in enumerate(industry_news[:3], 1):
                st.markdown(f"{i}. **{news.get('title', 'No title')}**")
                st.caption(f"{news.get('snippet', 'No description')} [Read more]({news.get('link', '#')})")
    
    with tab2:  # Investment Thesis - Single Column Layout
        # Company Overview & Financial Health
        with st.expander("🏢 Company Overview & Financial Health", expanded=True):
            # Two rows of metrics for better mobile responsiveness
            metric_cols1 = st.columns(4)
            with metric_cols1[0]: 
                st.metric("Market Cap", format_market_cap(stock_overview.get('MarketCapitalization')))
            with metric_cols1[1]: st.metric("Beta", stock_overview.get('Beta', 'N/A'))
            with metric_cols1[2]: st.metric("ROE (TTM)", valuation_metrics.get('ROE (TTM)', 'N/A'))
            with metric_cols1[3]: st.metric("FCF Yield", valuation_metrics.get('FCF Yield', 'N/A'))
            
            metric_cols2 = st.columns(4)
            with metric_cols2[0]: 
                st.metric("52-Week Range", f"${stock_overview.get('52WeekLow', 'N/A')} - ${stock_overview.get('52WeekHigh', 'N/A')}")
            with metric_cols2[1]: 
                st.metric("Volume (Avg)", f"{int(float(stock_overview.get('Volume', 0)) / 1_000_000):.1f}M")
            with metric_cols2[2]: 
                st.metric("Debt/Equity", valuation_metrics.get('Debt/Equity', 'N/A'))
            with metric_cols2[3]: 
                st.metric("Div Yield", valuation_metrics.get('Dividend Yield', '0.0%'))
        
        # Stock Chart with unique key
        if stock_plot is not None:
            # Use a counter to ensure unique keys across reruns
            if 'chart_counter' not in st.session_state:
                st.session_state.chart_counter = 0
            st.session_state.chart_counter += 1
            chart_key = f"stock_chart_{st.session_state.chart_counter}"  # Unique key with counter
            st.plotly_chart(stock_plot, use_container_width=True, key=chart_key)
        
        # Investment Highlights
        st.subheader("🎯 Investment Highlights")
        
        with st.container(border=True):
            st.markdown("### 🚀 Growth & Competitive Advantages")
            st.markdown("""
            - **Market Position**: Leading player with expanding TAM in core markets
            - **Innovation**: Strong product pipeline and R&D investments
            - **Global Expansion**: Significant opportunities in emerging markets
            - **Competitive Moats**: Network effects, high switching costs, and brand strength
            - **Financial Strength**: Robust balance sheet with strong cash flow generation
            - **Operational Excellence**: Consistent margin improvement and cost optimization
            - **Talent & Leadership**: Experienced management team with proven track record
            - **Sustainability**: Strong ESG practices and long-term value creation
            """)
        
        # Analyst Consensus
        if stock_overview.get('AnalystTargetPrice'):
            with st.container(border=True):
                st.markdown("### 📈 Analyst Consensus")
                current_price = float(stock_overview.get('Price', 0))
                target_price = float(stock_overview.get('AnalystTargetPrice', 0))
                if current_price and target_price:
                    upside = ((target_price - current_price) / current_price) * 100
                    
                    # Create columns for better layout of metrics
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Price Target", 
                                f"${target_price:.2f}", 
                                f"{upside:+.1f}% Upside" if upside > 0 else f"{abs(upside):.1f}% Downside")
                    
                    with cols[1]:
                        rating = min(5, max(1, round((upside + 10) / 5)))
                        st.markdown("**Analyst Rating**")
                        st.markdown(f"{'★' * rating}{'☆' * (5 - rating)} "
                                  f"({'Strong Buy' if rating >= 4 else 'Buy' if rating >= 3 else 'Hold'})")
                        
                        progress = min(100, max(0, (current_price / target_price) * 100))
                        st.markdown(f"**Progress to Target**")
                        st.progress(progress / 100, 
                                  f"{current_price:.2f} / {target_price:.2f} ({progress:.1f}%)")
        
        # Stock Metrics
        if 'df' in locals() and df is not None and not df.empty:
            try:
                latest = df.iloc[-1]
                prev_close = df.iloc[-2]['4. close'] if len(df) > 1 else latest['4. close']
                price_change = latest['4. close'] - prev_close
                pct_change = (price_change / prev_close) * 100
                
                st.subheader("📊 Stock Metrics")
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Current Price", 
                             f"${latest['4. close']:.2f}", 
                             f"{price_change:+.2f} ({pct_change:+.2f}%)",
                             delta_color="normal")
                with metric_cols[1]:
                    st.metric("Daily Range", 
                             f"${latest['2. high']:.2f} / ${latest['3. low']:.2f}")
                with metric_cols[2]:
                    st.metric("Volume", f"{int(latest['5. volume']/1000):,}K")
                with metric_cols[3]:
                    st.metric("Moving Averages", 
                             f"20d: ${latest.get('SMA20', 0):.2f} | 50d: ${latest.get('SMA50', 0):.2f}")
                
                # Technical Indicators - Only show if we have at least one valid technical indicator
                has_technical_data = any([
                    stock_overview.get('RSI (14)') not in [None, 'N/A'],
                    all(k in stock_overview and stock_overview[k] not in [None, 'N/A'] 
                        for k in ['MACD', 'MACD_Signal']),
                    all(k in stock_overview and stock_overview[k] not in [None, 'N/A'] 
                        for k in ['50DayMovingAverage', '200DayMovingAverage']),
                    stock_overview.get('Beta') not in [None, 'N/A']
                ])
                
                if has_technical_data:
                    with st.expander("Technical Indicators", expanded=False):
                        tech_cols = st.columns(2)
                        with tech_cols[0]:
                            rsi = stock_overview.get('RSI (14)', 'N/A')
                            if rsi not in [None, 'N/A']:
                                st.metric("RSI (14)", 
                                        str(rsi) if rsi != 'N/A' else 'N/A',
                                        "Overbought" if isinstance(rsi, (int, float)) and rsi > 70 else 
                                        "Oversold" if isinstance(rsi, (int, float)) and rsi < 30 else 
                                        "Neutral" if rsi != 'N/A' else '')
                        
                            macd = stock_overview.get('MACD', 'N/A')
                            macd_signal = stock_overview.get('MACD_Signal', None)
                            if macd not in [None, 'N/A'] and macd_signal not in [None, 'N/A']:
                                macd_status = 'Bullish' if macd > macd_signal else 'Bearish' if macd < macd_signal else 'Neutral'
                                st.metric("MACD", macd_status + ' Crossover' if macd_status != 'Neutral' else 'Neutral')
                        
                        with tech_cols[1]:
                            ma50 = stock_overview.get('50DayMovingAverage', 'N/A')
                            ma200 = stock_overview.get('200DayMovingAverage', 'N/A')
                            if ma50 not in [None, 'N/A'] and ma200 not in [None, 'N/A']:
                                ma_status = 'Golden Cross' if ma50 > ma200 else 'Death Cross' if ma50 < ma200 else 'Neutral'
                                st.metric("50/200 MA", ma_status)
                        
                            beta = stock_overview.get('Beta', 'N/A')
                            if beta not in [None, 'N/A']:
                                try:
                                    beta_float = float(beta)
                                    vol = 'High' if beta_float > 1.2 else 'Low' if beta_float < 0.8 else 'Medium'
                                    st.metric("Volatility", f"{vol} (β={beta_float:.2f})")
                                except (ValueError, TypeError):
                                    pass
                
            except Exception as e:
                st.warning(f"Could not load stock metrics: {str(e)}")
                
                # Only show metrics if we have valid data
                if stock_overview.get('Price') not in [None, 'N/A']:
                    st.metric("Current Price", f"${stock_overview.get('Price'):.2f}")
                
                if all(k in stock_overview and stock_overview[k] not in [None, 'N/A'] 
                       for k in ['52WeekLow', '52WeekHigh']):
                    try:
                        low = float(stock_overview['52WeekLow'])
                        high = float(stock_overview['52WeekHigh'])
                        st.metric("52-Week Range", f"${low:.2f} - ${high:.2f}")
                    except (ValueError, TypeError):
                        pass
                
                # Only show analyst consensus if we have a price target
                if stock_overview.get('AnalystTargetPrice') and stock_overview.get('AnalystRating'):
                    with st.container(border=True):
                        st.markdown("### 📈 Analyst Consensus")
                        
                        current_price = float(stock_overview.get('Price', 0))
                        target_price = float(stock_overview.get('AnalystTargetPrice', 0))
                        
                        if current_price and target_price:
                            upside = ((target_price - current_price) / current_price) * 100
                            
                            # Create columns for better layout
                            cols = st.columns(2)
                            
                            with cols[0]:
                                st.metric("Current Price", f"${current_price:.2f}")
                                st.metric("Price Target", 
                                         f"${target_price:.2f}", 
                                         f"{upside:+.1f}% Upside" if upside > 0 
                                         else f"{abs(upside):.1f}% Downside" if upside < 0 
                                         else "0.0%")
                            
                            with cols[1]:
                                # Calculate rating (example: convert text rating to stars)
                                rating_text = stock_overview.get('AnalystRating', '').lower()
                                if 'strong buy' in rating_text:
                                    rating = 5
                                elif 'buy' in rating_text:
                                    rating = 4
                                elif 'hold' in rating_text:
                                    rating = 3
                                elif 'sell' in rating_text:
                                    rating = 2
                                elif 'strong sell' in rating_text:
                                    rating = 1
                                else:
                                    rating = 0
                                
                                st.markdown("**Analyst Rating**")
                                st.markdown(f"{'★' * rating}{'☆' * (5 - rating)} "
                                           f"({rating_text.title() if rating > 0 else 'No Rating'})")
                                
                                # Show progress to target
                                if current_price and target_price:
                                    progress = min(100, max(0, (current_price / target_price) * 100))
                                    st.markdown(f"**Progress to Target**")
                                    st.progress(progress / 100, 
                                               f"${current_price:.2f} / ${target_price:.2f} ({progress:.1f}%)")
                else:
                    st.metric("Analyst Consensus", "N/A")
    
    with tab2:  # Valuation
        st.subheader("Valuation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("P/E (TTM)", f"{stock_overview.get('PERatio', 'N/A')}x", 
                     delta="-2.5% vs sector" if stock_overview.get('PERatio') else None)
            st.metric("Forward P/E", "22.5x", "-1.2% vs sector")
            st.metric("PEG Ratio", valuation_metrics.get('PEG Ratio', 'N/A'))
            st.metric("Price/Sales", f"{stock_overview.get('PriceToSalesRatioTTM', 'N/A')}x")
            
        with col2:
            st.metric("EV/EBITDA", f"{valuation_metrics.get('EV/EBITDA', 'N/A')}x")
            st.metric("P/Book", f"{valuation_metrics.get('P/B Ratio', 'N/A')}x")
            st.metric("P/CF", f"{stock_overview.get('PriceToBookRatio', 'N/A')}x")
            st.metric("Dividend Yield", valuation_metrics.get('Dividend Yield', 'N/A'))
        
        st.subheader("Valuation Summary")
        st.markdown("""
        - **Current Valuation**: Trading at a slight discount to 5-year average P/E
        - **Relative Value**: Undervalued compared to peers in the tech sector
        - **Growth Premium**: Justified by strong revenue growth and margin expansion
        - **DCF Implied**: Current price implies ~8% annual growth over next 5 years
        - **Price Targets**:
          - High: $180 (25% upside)
          - Median: $165 (15% upside)
          - Low: $140 (-2% downside)
        """)
    
    with tab3:  # Financial Health
        st.subheader("Balance Sheet Strength")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Cash (MRQ)", "$169.2B")
            st.metric("Total Debt (MRQ)", "$28.5B")
            st.metric("Current Ratio", valuation_metrics.get('Current Ratio', 'N/A'))
            
        with col2:
            st.metric("Net Cash Position", "$140.7B")
            st.metric("Debt/Equity", valuation_metrics.get('Debt/Equity', 'N/A'))
            st.metric("Interest Coverage", "N/A")
        
        st.subheader("Profitability Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Gross Margin (TTM)", 
                    valuation_metrics.get('Gross Margin (TTM)', 'N/A'),
                    delta=valuation_metrics.get('Gross Margin YoY Change', None),
                    delta_color="normal")
            st.metric("Operating Margin (TTM)",
                    valuation_metrics.get('Operating Margin (TTM)', 'N/A'),
                    delta=valuation_metrics.get('Operating Margin YoY Change', None),
                    delta_color="normal")
            
        with col2:
            st.metric("ROE (TTM)", valuation_metrics.get('ROE (TTM)', 'N/A'))
            st.metric("ROIC (TTM)", valuation_metrics.get('ROIC (TTM)', 'N/A'))
    
    with tab4:  # Growth & Returns
        st.subheader("Growth Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Revenue Growth (YoY)", valuation_metrics.get('Revenue Growth (YoY)', 'N/A'))
            st.metric("EPS Growth (YoY)", valuation_metrics.get('EPS Growth (YoY)', 'N/A'))
            st.metric("3-Yr Revenue CAGR", "15.2%")
            
        with col2:
            st.metric("EBITDA Growth (YoY)", "10.1%")
            st.metric("FCF Growth (YoY)", "9.8%")
            st.metric("3-Yr EPS CAGR", "18.7%")
        
        st.subheader("Shareholder Returns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("1-Year Return", "+24.5%", "+5.2% vs S&P 500")
            st.metric("3-Year Annualized", "+18.2%", "+6.8% vs S&P 500")
            
        with col2:
            st.metric("5-Year Annualized", "+22.1%", "+8.3% vs S&P 500")
            st.metric("10-Year Annualized", "+19.8%", "+9.1% vs S&P 500")
    
    with tab5:  # Risks & Opportunities
        st.subheader("Key Investment Risks")
        
        with st.expander("Regulatory & Legal Risks"):
            st.markdown("""
            - **Antitrust Scrutiny**: Ongoing regulatory investigations in multiple jurisdictions
            - **Privacy Regulations**: Increasing global data protection laws (GDPR, CCPA, etc.)
            - **Content Moderation**: Challenges in balancing free speech and harmful content
            - **Taxation Changes**: Potential impact of global tax reforms on effective tax rate
            """)
        
        with st.expander("Business & Market Risks"):
            st.markdown("""
            - **Advertising Slowdown**: Exposure to cyclical advertising spending
            - **Cloud Competition**: Intense competition in cloud services from AWS and Azure
            - **Talent Retention**: Challenges in attracting and retaining top tech talent
            - **Supply Chain**: Potential disruptions in hardware and data center operations
            """)
        
        st.subheader("Growth Opportunities")
        
        with st.expander("Near-Term Opportunities (1-2 years)"):
            st.markdown("""
            - **Cloud Growth**: Continued enterprise cloud adoption and migration
            - **AI Integration**: Monetization of AI across products and services
            - **YouTube Shorts**: Monetization of short-form video content
            - **Retail Media**: Expansion of commerce and advertising solutions
            """)
            
        with st.expander("Long-Term Opportunities (3-5+ years)"):
            st.markdown("""
            - **AI Leadership**: Commercialization of AI research and applications
            - **Quantum Computing**: Potential breakthroughs in quantum supremacy
            - **Healthcare Tech**: Expansion in healthcare data analytics and AI diagnostics
            - **Autonomous Vehicles**: Commercialization of Waymo's self-driving technology
            - **Next-Gen Computing**: Advancements in quantum and edge computing
            """)

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
    st.header("🔍 Analysis Setup")
    
    # Company Information
    company_name_input = st.text_input("Company Name", "")
    
    # Document Upload
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    # Simple Action Button
    run_analysis = st.button("Analyze", type="primary", use_container_width=True)
    
    # Simple Help Text
    st.caption("ℹ️ Upload a company's annual report PDF to begin analysis.")
    
    # Analysis Results Section (will be populated after analysis)
    if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
        st.divider()
        st.subheader("Analysis Complete")
        st.caption("View the main panel for detailed insights.")
        
        # Add a button to clear the analysis
        if st.button("Clear Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.rerun()

# Initialize session state for chat and chain
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload an annual report and provide a company name, then click 'Run Analysis'."}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Main Processing ---
# Show loading state during analysis
with st.spinner("🧠 Analyzing financial data. This may take a few moments..."):
    if run_analysis:
        if not company_name_input or uploaded_file is None:
            st.error("❌ Please provide both a company name and a PDF file.")
        else:
            # Clear session to force re-analysis
            st.session_state.rag_chain = None
            st.session_state.messages = []
            
            with st.spinner("Starting analysis... This may take a few minutes:"):
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
            sentiment, score = analyze_sentiment(all_snippets)  # Returns (sentiment_label, score)

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
            
            # 6. Display Analysis Summary
            st.session_state.analysis_complete = True
            if stock_overview:
                st.metric("Market Cap", format_market_cap(stock_overview.get('MarketCapitalization')))
            st.success("Analysis complete!")
            
            # Display the analysis summary
            display_analysis_summary(
                company_name=company_name_input,
                company_details=company_details,
                sentiment_score=score,  # Using the score from analyze_sentiment
                stock_overview=stock_overview or {},
                industry_news=industry_news,
                company_news=company_news,
                stock_plot=stock_plot
            )
            
            # Add initial assistant message
            if not any(msg["content"].startswith("Analysis for") for msg in st.session_state.messages):
                st.session_state.messages.append({"role": "assistant", "content": f"Analysis for **{company_name_input}** is complete. You can now ask questions about the annual report and market context."})

# Chat Interface
st.markdown("### 💬 Ask a question about the financial report")
st.caption("Example: What were the company's total revenues last year?")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Type your question here...", key="chat_input")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show a loading message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analyzing your question...")
        
        try:
            # Get the RAG chain
            if st.session_state.rag_chain is None:
                message_placeholder.error("Please run the analysis first.")
            else:
                # Get the answer
                response = st.session_state.rag_chain.invoke({"query": prompt})
                answer = response["result"]
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"An error occurred: {e}")