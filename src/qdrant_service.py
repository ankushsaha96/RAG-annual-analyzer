"""Qdrant VectorDB service + NSE annual report fetching module."""

import os
import re
import uuid
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import quote

import requests
import numpy as np
import faiss
import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from qdrant_client.http.models import PayloadSchemaType

from src.config import get_config

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────
BATCH_SIZE = 128
CHUNK_SIZE = 20  # sentences per chunk
UPSERT_BATCH = 100


class QdrantService:
    """Wraps Qdrant client for annual report embeddings."""

    def __init__(self):
        config = get_config()
        self.collection = config.qdrant.collection_name
        self.vector_size = config.qdrant.vector_size

        self.client = QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key,
        )

        # NVIDIA embedding client
        api_key = config.embedding.nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.embed_client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        self.embed_model = config.embedding.model_name

        # Ensure collection exists
        self._ensure_collection()

    # ── Collection Setup ──────────────────────────────────
    def _ensure_collection(self):
        """Create collection + payload indices if they don't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            logger.info(f"Creating Qdrant collection '{self.collection}'...")
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            )
            # Payload indices for filtered search
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="company",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="year",
                field_schema=PayloadSchemaType.INTEGER,
            )
            logger.info("Collection and indices created.")

    # ── Public API ────────────────────────────────────────
    def check_embeddings_exist(self, company_name: str, year: int) -> bool:
        """Return True if embeddings for company+year exist in Qdrant."""
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="company", match=MatchValue(value=company_name)),
                    FieldCondition(key="year", match=MatchValue(value=year)),
                ]
            ),
            limit=1,
        )
        return len(records) > 0

    def create_embeddings(self, company_name: str, year: int) -> dict:
        """
        Full pipeline: fetch PDF → extract → chunk → embed → upsert.
        Returns a status dict.
        """
        # Double-check existence
        if self.check_embeddings_exist(company_name, year):
            return {"status": "exists", "message": f"Data for {company_name} {year} already exists."}

        # 1. Fetch the annual report PDF
        logger.info(f"Fetching annual report for {company_name} {year}...")
        pdf_path = fetch_annual_report(company_name, year, output_dir="Data")
        if pdf_path is None:
            raise RuntimeError(f"Could not fetch annual report for {company_name} {year}")

        # 2. Extract text
        logger.info(f"Extracting text from {pdf_path}...")
        pages_and_texts = self._open_and_read_pdf(pdf_path)

        # 3. Process page-by-page (low RAM)
        total_indexed = 0
        for item in pages_and_texts:
            page_num = item["page_number"]
            raw_text = item.get("text", "")

            # Split into sentences
            sentences = [s.strip() + "." for s in raw_text.split(". ") if s.strip()]
            if not sentences:
                continue

            # Chunk sentences
            chunks = _split_list(sentences, CHUNK_SIZE)

            # Process in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_raw = chunks[i: i + BATCH_SIZE]
                batch_texts = [" ".join(c).replace("  ", " ").strip() for c in batch_raw]

                try:
                    responses = self.embed_client.embeddings.create(
                        input=batch_texts,
                        model=self.embed_model,
                        encoding_format="float",
                        extra_body={"input_type": "passage", "truncate": "END"},
                    ).data

                    batch_points = []
                    for text, resp in zip(batch_texts, responses):
                        embed = np.array(resp.embedding).astype("float32").reshape(1, -1)
                        faiss.normalize_L2(embed)

                        batch_points.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embed[0].tolist(),
                                payload={
                                    "company": company_name,
                                    "year": year,
                                    "document_type": "annual_report",
                                    "text": text,
                                    "page": page_num,
                                },
                            )
                        )

                    # Upsert immediately to free RAM
                    self.client.upsert(
                        collection_name=self.collection,
                        points=batch_points,
                    )
                    total_indexed += len(batch_points)
                except Exception as e:
                    logger.error(f"Error on page {page_num} batch {i}: {e}")
                    continue

        logger.info(f"✅ Successfully indexed {total_indexed} chunks for {company_name} {year} to Qdrant")

        # 4. Clean up — delete the downloaded PDF to save disk
        try:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"🗑️ Deleted PDF: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not delete PDF {pdf_path}: {e}")

        return {"status": "created", "message": f"Indexed {total_indexed} chunks.", "chunks": total_indexed}

    def query_points(
        self,
        query_embedding: np.ndarray,
        company_name: str,
        year: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Filtered vector search in Qdrant."""
        # Normalize
        qe = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(qe)
        qe = qe[0]

        logger.info(f"🔍 Querying Qdrant collection '{self.collection}' for company='{company_name}', year={year}")

        results = self.client.query_points(
            collection_name=self.collection,
            query=qe.tolist(),
            query_filter={
                "must": [
                    {"key": "company", "match": {"value": company_name}},
                    {"key": "year", "match": {"value": year}},
                ]
            },
            limit=limit,
        )

        logger.info(f"📦 Retrieved {len(results.points)} chunks from Qdrant")

        retrieved = []
        for r in results.points:
            retrieved.append({
                "sentence_chunk": r.payload.get("text", ""),
                "text": r.payload.get("text", ""),
                "page_number": r.payload.get("page", -1),
                "score": float(r.score),
            })

        return retrieved

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into an embedding vector."""
        response = self.embed_client.embeddings.create(
            input=[query],
            model=self.embed_model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        return np.array(response.data[0].embedding).astype("float32")

    # ── Internal helpers ──────────────────────────────────
    @staticmethod
    def _open_and_read_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        """Read PDF and return page dicts. Adapted from notebook Cell 2."""
        doc = fitz.open(pdf_path)
        pages = []
        for page_number, page in enumerate(doc):
            text = page.get_text().replace("\n", " ").strip()
            pages.append({
                "page_number": page_number,
                "text": text,
            })
        return pages


def _split_list(input_list: list, slice_size: int) -> list:
    """Split list into sub-lists of slice_size."""
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


# ══════════════════════════════════════════════════════════
#  NSE Annual Report Fetcher (from notebook Cell 1)
# ══════════════════════════════════════════════════════════

def fetch_annual_report(company_name: str, year: int, output_dir: str = ".") -> Optional[str]:
    """
    Download the annual report PDF for an Indian listed company from NSE.

    Parameters
    ----------
    company_name : str   — e.g. "TCS", "Infosys", "Reliance Industries"
    year : int           — FY END year.  For FY 2023-24, pass 2024.
    output_dir : str     — Directory where the PDF will be saved.

    Returns
    -------
    str | None — Absolute path to the downloaded PDF, or None if not found.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]", "_", company_name)
    dest = os.path.join(output_dir, f"{safe_name}_AnnualReport_{year}.pdf")
    fy = f"{year - 1}-{str(year)[-2:]}"

    # Check if already downloaded
    if os.path.exists(dest):
        logger.info(f"PDF already exists at {dest}")
        return os.path.abspath(dest)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Origin": "https://www.nseindia.com",
    })

    # Step 1: warm up session
    logger.info("Connecting to NSE India...")
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception as e:
        logger.warning(f"Could not reach NSE homepage: {e}")

    # Step 2: search for company
    logger.info(f"Searching NSE for '{company_name}'...")
    try:
        session.get("https://www.nseindia.com", timeout=10)
        session.get("https://www.nseindia.com/market-data/all-reports", timeout=10)

        search_url = f"https://www.nseindia.com/api/search/autocomplete?q={quote(company_name)}"
        r = session.get(search_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        symbols = data.get("symbols", [])
        candidates = [s for s in symbols if s.get("symbol_info", "").upper() == "EQ"
                      or s.get("result_type", "") == "symbol"]
        if not candidates:
            candidates = symbols
    except Exception as e:
        logger.error(f"NSE search failed: {e}")
        return None

    if not candidates:
        logger.error(f"No company found for '{company_name}'")
        return None

    query_upper = company_name.upper().strip()

    def name_score(item):
        symbol = str(item.get("symbol", "")).upper().strip()
        name = str(item.get("symbol_info", "")).upper().strip()
        subtype = str(item.get("result_sub_type", "")).lower()
        s = 0
        if subtype == "equity": s += 5
        if symbol == query_upper: s += 10
        elif symbol.startswith(query_upper): s += 6
        elif query_upper in symbol: s += 3
        if query_upper in name: s += 2
        return s

    best = sorted(candidates, key=name_score, reverse=True)[0]
    nse_symbol = str(best.get("symbol", "")).strip()

    if not nse_symbol:
        logger.error(f"Could not extract NSE symbol from: {best}")
        return None

    logger.info(f"Selected NSE symbol: {nse_symbol}")

    # Step 3: fetch annual report filings
    logger.info(f"Fetching annual report filings for FY {fy}...")
    filings_url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={quote(nse_symbol)}"
    try:
        r = session.get(filings_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        filings = data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.error(f"Could not fetch filings: {e}")
        return None

    if not filings:
        logger.error(f"No annual report filings found for {nse_symbol}")
        return None

    # Step 4: score and pick best match
    def score(item):
        from_yr = str(item.get("fromYr", ""))
        to_yr = str(item.get("toYr", ""))
        s = 0
        s += 4 if from_yr == str(year - 1) and to_yr == str(year) else 0
        s += 2 if from_yr == str(year - 1) else 0
        s += 1 if to_yr == str(year) else 0
        return s

    chosen = sorted(filings, key=score, reverse=True)[0]
    file_link = chosen.get("fileName", "")

    if not file_link:
        logger.error("No download link found in filing.")
        return None

    if file_link.startswith("/"):
        file_link = "https://www.nseindia.com" + file_link

    # Step 5: download the PDF
    logger.info(f"Downloading from: {file_link}")
    try:
        r = session.get(file_link, timeout=60, stream=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            logger.error("Server returned HTML instead of PDF — session blocked.")
            return None

        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        # Check for ZIP
        with open(dest, "rb") as f:
            header = f.read(4)

        if header == b"PK\x03\x04":
            logger.info("File is a ZIP archive — extracting PDF...")
            import zipfile, shutil
            zip_path = dest + ".zip"
            shutil.move(dest, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                pdf_files = [n for n in z.namelist() if n.lower().endswith(".pdf")]
                if not pdf_files:
                    os.remove(zip_path)
                    logger.error("No PDF found inside ZIP.")
                    return None
                pdf_files.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
                chosen_pdf = pdf_files[0]
                pdf_bytes = z.read(chosen_pdf)
            os.remove(zip_path)
            with open(dest, "wb") as f:
                f.write(pdf_bytes)

        elif header != b"%PDF":
            os.remove(dest)
            logger.error("Downloaded file is not a valid PDF.")
            return None

        logger.info(f"Downloaded {downloaded / (1024 * 1024):.2f} MB to {dest}")
        return os.path.abspath(dest)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None
