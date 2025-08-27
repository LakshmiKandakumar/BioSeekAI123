import subprocess
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import requests
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------- CONFIG ----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "pritamdeka/S-PubMedBERT-MS-MARCO"
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

CANDIDATE_RETMAX = 100  # fetch top 100 articles from PubMed
TOP_N = 10
ABSTRACT_TRUNCATE = None  # full abstract
import os
from dotenv import load_dotenv

load_dotenv("key.env")

API_KEY = os.getenv("PUBMED_API_KEY")



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


# ---------------- OLLAMA + MESH EXTRACTION ----------------

def extract_topic_with_ollama(user_query: str) -> str:
    """
    Uses local Ollama to extract a main biomedical topic and map it to MeSH.
    """
    prompt = f"""
    You are a biomedical research assistant.
    Extract only the main biomedical topic from this user query.
    Output a term that matches MeSH vocabulary if possible.
    Do not include other phrases or sentences.
    Query: "{user_query}"
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],  # replace with your local model
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        topic = result.stdout.strip()
    except FileNotFoundError as e:
        print("Ollama not found, falling back to user query.")
        topic = user_query  # fallback
    except subprocess.CalledProcessError as e:
        print("Error calling Ollama:", e)
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


# ---------------- SEMANTIC SEARCH ----------------

def semantic_search(user_query: str, top_n=TOP_N):
    # extract topic
    topic_query = extract_topic_with_ollama(user_query)

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

    # encode query and abstracts in batch
    query_emb = model.encode(topic_query, convert_to_tensor=True, device=DEVICE)
    doc_embs = model.encode(abstracts, convert_to_tensor=True, device=DEVICE, batch_size=32, show_progress_bar=False)

    sims = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    for i, a in enumerate(articles):
        a["similarity"] = float(sims[i])

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
