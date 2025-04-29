import streamlit as st
import pandas as pd
import io
from ragpart import generate_response_from_chunks, get_relevant_chunks, create_index, extract_text_from_pdf, clean_text, store_chunks_in_pinecone, combined_chunking
from translate import translate, generate_audio
from arxiv import process_docs2
import os

os.environ["TRANSFORMERS_CACHE"] = r"C:\m2m100_418M"    

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# For translation

def translate(text, lang):
    try:
        # Tokenize the input text
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate the translation
        output = model.generate(**tokens, forced_bos_token_id=tokenizer.get_lang_id(lang))
        
        # Decode the output
        translated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        st.error(f"An error occurred during translation: {e}")
        return ""


# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'search' not in st.session_state:
    st.session_state.search = []
if 'query' not in st.session_state:
    st.session_state.query = None
if 'download' not in st.session_state:
    st.session_state.download = False
if 'papers_downloaded' not in st.session_state:
    st.session_state.papers_downloaded = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []
if 'cluster' not in st.session_state:
    st.session_state.cluster = None

def reset_page():
    st.session_state.index = None
    st.session_state.search = []
    st.session_state.query = None
    st.session_state.papers_downloaded = False
    st.session_state.result_df = None
    st.session_state.fig = None
    st.session_state.selected_cluster = None
    st.session_state.selected_indices = []
    st.session_state.cluster = None

def preload_pdf(file_path):
    with open(file_path, "rb") as f:
        return io.BytesIO(f.read())

def process_local_pdfs(data):
    combined_chunks = []
    
    # Check if data is a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict()
        data = data['text']

    # If data is a list of uploaded files
    for pdf_file in data:
        if isinstance(data, dict) and isinstance(data[pdf_file], str):
            text = data[pdf_file]  
        else:
            text = extract_text_from_pdf(pdf_file)
        
        cleaned_text = clean_text(text)
        chunks = combined_chunking(cleaned_text)
        combined_chunks.extend(chunks)
    
    return combined_chunks

def handle_query_response(query, lang):
    relevant_chunks = get_relevant_chunks(query, st.session_state.index)
    response = generate_response_from_chunks(relevant_chunks, query)
    if lang == "French":
        translated_response = translate(response, "fr")
        st.write(translated_response)
    else:
        st.write(response)
    audio_io = generate_audio(response, lang)
    st.audio(audio_io, format='audio/mp3')
    st.download_button(label="Download Audio Response", data=audio_io, file_name="response.mp3", mime="audio/mp3")

# Streamlit app
st.sidebar.image("logo.jpg")
st.title("Commercial Courts Research Engine")
st.sidebar.title("PDF Research Assistant")

lang = st.sidebar.radio("Choose", ["English", "French"])

# Language map
language_map = {
    'English': 'en-US',
    'French': 'fr-FR'
}

# Preload PDF file (set path to your preloaded PDF)
pdf_file_path = "commercialcourtsact2015.pdf"
preloaded_pdf = preload_pdf(pdf_file_path)

# Use the preloaded PDF
data = [preloaded_pdf] if preloaded_pdf else []

if data and not st.session_state.papers_downloaded:
    with st.spinner("Processing PDFs..."):
        combined_chunks = process_local_pdfs(data)
        st.session_state.index = create_index()
        if st.session_state.index:
            store_chunks_in_pinecone(combined_chunks, st.session_state.index)
            st.session_state.papers_downloaded = True
            st.success("PDF processed and indexed successfully!")
        else:
            st.error("Failed to create Pinecone index.")

# Query handling
if st.session_state.index:
    query = st.text_input("Enter your question:")
    if query:
        st.session_state.query = query
    if st.button("Ask") and st.session_state.query:
        with st.spinner("Searching for answers..."):
            handle_query_response(st.session_state.query, lang)
        
    if st.button("End conversation"):
        reset_page()
        st.rerun()