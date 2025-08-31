# biomed_search.py
import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from sentence_transformers import util
import torch

# ---------------- CONFIG ----------------
DEVICE = "cpu"  # Azure App Service CPU
CANDIDATE_RETMAX = 100
TOP_N = 10
ABSTRACT_WORD_LIMIT = 100

# Load environment variables
load_dotenv("key.env")
API_KEY = os.getenv("PUBMED_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------- Lazy-load model & LLM client ----------------
_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("pritamdeka/S-PubMedBERT-MS-MARCO", device=DEVICE)
    return _model

def get_llama_client():
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(token=HF_TOKEN)
    return _client

# ---------------- UTILITY ----------------
def truncate_abstract_words(text: str, max_words: int = ABSTRACT_WORD_LIMIT) -> str:
    if not text:
        return "No abstract"
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "..."

def map_to_mesh(term: str) -> str | None:
    """Map term to MeSH descriptor using NCBI E-utilities."""
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "mesh", "term": term, "retmode": "json", "retmax": 1, "api_key": API_KEY}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "mesh", "id": ids[0], "retmode": "xml", "api_key": API_KEY}
        res = requests.get(fetch_url, params=fetch_params, timeout=10)
        root = ET.fromstring(res.content)
        return root.findtext(".//DescriptorName")
    except Exception:
        return None

# ---------------- LLM via HF API ----------------
def extract_topic_with_llama_hf(user_query: str) -> tuple[str, str]:
    """Use HF Inference API to extract main topic & expanded query."""
    client = get_llama_client()
    system_message = "You are a biomedical research assistant. Provide main topic and expanded query."
    prompt = f"""
User Query: "{user_query}"
Main Topic: [single most important biomedical concept]
Expanded Query: [enhanced version with synonyms and MeSH-compatible terms]
"""
    try:
        response = client.chat_completion(
            model="meta-llama/3b-instruct",  # lighter model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        llm_output = response["choices"][0]["message"]["content"].strip() if response and "choices" in response else ""
        main_topic, expanded_query = user_query, user_query
        for line in llm_output.splitlines():
            if line.lower().startswith("main topic:"):
                main_topic = line.split(":", 1)[1].strip()
            elif line.lower().startswith("expanded query:"):
                expanded_query = line.split(":", 1)[1].strip()
        return expanded_query, main_topic
    except Exception:
        return user_query, user_query

# ---------------- PUBMED FUNCTIONS ----------------
def build_hybrid_query(expanded_query: str, extracted_topic: str) -> str:
    mesh_term = map_to_mesh(extracted_topic)
    query_blocks = [f'"{mesh_term}"[MeSH Terms]'] if mesh_term else []
    query_blocks.append(f"({expanded_query})")
    return " OR ".join(query_blocks)

def esearch_pubmed(query: str, retmax=CANDIDATE_RETMAX):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax,
              "api_key": API_KEY, "sort": "relevance"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])

def efetch_chunk(pmids_chunk):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids_chunk), "retmode": "xml", "api_key": API_KEY}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.content

def parse_pubmed_article(article):
    title = article.findtext(".//ArticleTitle", "No title")
    authors = ", ".join([f"{a.findtext('ForeName','')} {a.findtext('LastName','')}"
                         for a in article.findall(".//Author")
                         if a.findtext("LastName") and a.findtext("ForeName")]) or "No authors listed"
    year_elem = article.find(".//PubDate/Year")
    medline_date = article.find(".//PubDate/MedlineDate")
    year = year_elem.text if year_elem is not None else (medline_date.text if medline_date is not None else "Unknown")
    abstract = " ".join([abst.text for abst in article.findall(".//AbstractText") if abst.text])
    abstract = truncate_abstract_words(abstract)
    pmid = article.findtext(".//PMID", "")
    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
    return {"title": title, "authors": authors, "year": year, "abstract": abstract, "link": link}

def efetch_pubmed_optimized(pmids):
    chunks = [pmids[i:i + 100] for i in range(0, len(pmids), 100)]
    def fetch_and_parse_chunk(chunk):
        xml_data = efetch_chunk(chunk)
        root = ET.fromstring(xml_data)
        return [parse_pubmed_article(a) for a in root.findall(".//PubmedArticle")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_and_parse_chunk, chunks)
    return [a for sublist in results for a in sublist]

# ---------------- MAIN PIPELINE ----------------
def semantic_search(user_query: str, top_n=TOP_N):
    expanded_query, extracted_topic = extract_topic_with_llama_hf(user_query)
    query_to_use = build_hybrid_query(expanded_query, extracted_topic)
    pmids = esearch_pubmed(query_to_use)
    if not pmids:
        return []

    articles = efetch_pubmed_optimized(pmids)
    unique_articles = {a["link"]: a for a in articles if a["abstract"] != "No abstract"}
    articles = list(unique_articles.values())
    if not articles:
        return []

    # Encode abstracts only when needed
    model = get_model()
    abstracts = [a["abstract"] for a in articles]
    query_emb = model.encode(extracted_topic, convert_to_tensor=True, device=DEVICE)
    doc_embs = model.encode(abstracts, convert_to_tensor=True, device=DEVICE, batch_size=16, show_progress_bar=False)
    sims = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    for i, a in enumerate(articles):
        a["similarity"] = float(sims[i])

    sorted_articles = sorted(articles, key=lambda x: x["similarity"], reverse=True)[:top_n]
    return [{"title": a["title"], "authors": a["authors"], "year": a["year"],
             "abstract": a["abstract"], "link": a["link"]} for a in sorted_articles]
