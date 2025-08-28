import subprocess
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------- CONFIG ----------------

load_dotenv("key.env")

API_KEY = os.getenv("PUBMED_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBys3qxTdGbxJFwXDPLnaP9VUcOO7fvCxU")
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini client initialized in Model.py")
except Exception as e:
    print(f"❌ Failed to initialize Gemini in Model.py: {e}")
    gemini_model = None

def _gemini_chat(messages):
    if not gemini_model:
        raise RuntimeError("Gemini client not initialized in Model.py")
    
    try:
        # Convert messages to simple prompt for Gemini
        if len(messages) == 1 and messages[0].get("role") == "user":
            prompt = messages[0]["content"]
        else:
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += msg["content"]
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.3,
            )
        )
        
        # Create response object similar to Groq format
        class GeminiResponse:
            def __init__(self, text):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': text})()
                })()]
        
        return GeminiResponse(response.text)
        
    except Exception as e:
        raise RuntimeError(f"Gemini API failed in Model.py: {e}")

CANDIDATE_RETMAX = 100  # fetch top 100 articles from PubMed
TOP_N = 10
ABSTRACT_TRUNCATE = None  # full abstract

# ---------------- UTILITIES ----------------
def truncate_text(text: str, max_chars: int = ABSTRACT_TRUNCATE) -> str:
    if text is None:
        return "No abstract"
    return text if max_chars is None else text[:max_chars]


# ---------------- MeSH VALIDATION ----------------
def validate_with_mesh(term: str) -> bool:
    """
    Check if a given term exists in MeSH database.
    Returns True if valid MeSH term, else False.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "mesh", "term": term, "retmode": "json", "retmax": 1, "api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        return bool(ids)
    except Exception as e:
        print("MeSH validation error:", e)
        return False


def map_to_mesh(term: str) -> str | None:
    """
    Maps a free-text term to a valid MeSH term using PubMed E-utilities.
    Returns None if no match is found.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "mesh", "term": term, "retmode": "json", "retmax": 1, "api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        # fetch the official MeSH term
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "mesh", "id": ids[0], "retmode": "xml", "api_key": API_KEY}
        res = requests.get(fetch_url, params=fetch_params, timeout=10)
        root = ET.fromstring(res.content)
        return root.findtext(".//DescriptorName")
    except Exception as e:
        print("Error mapping to MeSH:", e)
        return None


# ---------------- GEMINI + MESH EXTRACTION ----------------

def extract_topic_with_groq(user_query: str) -> str:
    """
    Uses Gemini API to extract a main biomedical topic and map it to MeSH.
    """
    prompt = f"""
    You are a biomedical research assistant.
    Extract only the main biomedical topic from this user query.
    Output a term that matches MeSH vocabulary if possible.
    Do not include other phrases or sentences.
    Query: "{user_query}"
    """

    try:
        response = _gemini_chat(
            messages=[{"role": "user", "content": prompt}]
        )
        topic = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error calling Gemini API:", e)
        topic = user_query  # fallback

    # Map to official MeSH term if available
    mesh_term = map_to_mesh(topic)
    return mesh_term or topic


# ---------------- PUBMED SEARCH ----------------

def esearch_pubmed(query: str, retmax=CANDIDATE_RETMAX):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "api_key": API_KEY,
        "sort": "relevance",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch_chunk(pmids_chunk):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids_chunk), "retmode": "xml", "api_key": API_KEY}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.content


# ---------------- PARSING ----------------

def parse_pubmed_article(article):
    title = article.findtext(".//ArticleTitle", "No title")
    authors = ", ".join([f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}"
                         for a in article.findall(".//Author") if a.findtext('LastName') and a.findtext('ForeName')])
    authors = authors or "No authors listed"

    year_elem = article.find(".//PubDate/Year")
    medline_date = article.find(".//PubDate/MedlineDate")
    year = year_elem.text if year_elem is not None else (medline_date.text if medline_date is not None else "Unknown")

    abstract = " ".join([abst.text for abst in article.findall(".//AbstractText") if abst.text])
    abstract = truncate_text(abstract) or "No abstract"

    pmid = article.findtext(".//PMID", "")
    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

    return {"title": title, "authors": authors, "year": year, "abstract": abstract, "link": link}


def efetch_pubmed_optimized(pmids):
    chunks = [pmids[i:i+100] for i in range(0, len(pmids), 100)]  # fetch 100 PMIDs per request

    def fetch_and_parse_chunk(chunk):
        xml_data = efetch_chunk(chunk)
        root = ET.fromstring(xml_data)
        return [parse_pubmed_article(a) for a in root.findall(".//PubmedArticle")]

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(fetch_and_parse_chunk, chunks)

    return [a for sublist in results for a in sublist]


# ---------------- SEMANTIC SEARCH WITH HF API ----------------

def get_embeddings_from_hf(texts, model_name="pritamdeka/S-PubMedBERT-MS-MARCO"):
    """
    Get embeddings using Hugging Face Inference API
    """
    if isinstance(texts, str):
        texts = [texts]
    
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        response = requests.post(url, headers=headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting embeddings from HF API: {e}")
        return None


def semantic_search(user_query: str, top_n=TOP_N):
    # extract topic
    topic_query = extract_topic_with_groq(user_query)

    # ✅ validate topic
    if validate_with_mesh(topic_query):
        print(f"✅ '{topic_query}' is a valid MeSH term, using MeSH search...")
        query_to_use = f"{topic_query}[MeSH Terms]"
    else:
        print(f"⚠️ '{topic_query}' not found in MeSH, using as free-text...")
        query_to_use = topic_query

    pmids = esearch_pubmed(query_to_use)
    if not pmids:
        print("No articles found.")
        return []

    articles = efetch_pubmed_optimized(pmids)
    articles = [a for a in articles if a["abstract"] != "No abstract"]
    if not articles:
        print("No articles with abstracts found.")
        return []

    abstracts = [a["abstract"] for a in articles]

    # Get embeddings using HF API
    query_emb = get_embeddings_from_hf(topic_query)
    if query_emb is None:
        print("Failed to get query embedding, returning articles without similarity scores")
        return articles[:top_n]
    
    # Get embeddings for all abstracts
    doc_embs = get_embeddings_from_hf(abstracts)
    if doc_embs is None:
        print("Failed to get document embeddings, returning articles without similarity scores")
        return articles[:top_n]

    # Calculate similarities (assuming embeddings are returned as lists)
    if isinstance(query_emb, list) and len(query_emb) > 0:
        query_vector = query_emb[0] if isinstance(query_emb[0], list) else query_emb[0]
        
        for i, art in enumerate(articles):
            if i < len(doc_embs) and isinstance(doc_embs[i], list):
                doc_vector = doc_embs[i][0] if isinstance(doc_embs[i][0], list) else doc_embs[i][0]
                
                # Calculate cosine similarity
                import numpy as np
                query_norm = np.linalg.norm(query_vector)
                doc_norm = np.linalg.norm(doc_vector)
                
                if query_norm > 0 and doc_norm > 0:
                    similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
                    art["similarity"] = float(similarity)
                else:
                    art["similarity"] = 0.0
            else:
                art["similarity"] = 0.0
    else:
        # Fallback: assign random similarities
        import random
        for art in articles:
            art["similarity"] = random.uniform(0.1, 0.9)

    return sorted(articles, key=lambda x: x["similarity"], reverse=True)[:top_n]


# ---------------- TEST ----------------

if __name__ == "__main__":
    import time

    query = input("Enter your biomedical query: ")
    start_time = time.time()

    results = semantic_search(query)

    end_time = time.time()
    elapsed = end_time - start_time

    for i, art in enumerate(results, start=1):
        print(f"\n{i}. {art['title']} (sim={art['similarity']:.4f})")
        print(f"Year: {art['year']}")
        print(f"Authors: {art['authors']}")
        print(f"Link: {art['link']}")
        print(f"Abstract: {art['abstract']}\n")

    print(f"Search completed in {elapsed:.2f} seconds")
