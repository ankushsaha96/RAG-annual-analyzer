"""PDF extraction and text processing module."""

import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


def text_formatter(text: str) -> str:
    """
    Perform minor formatting on extracted text.
    
    Args:
        text: Raw text to format.
        
    Returns:
        Cleaned text with newlines replaced and stripped.
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str, page_offset: int = 4) -> List[Dict[str, Any]]:
    """
    Open a PDF file and extract text content page by page.
    
    This function reads through a PDF document, extracts text from each page,
    and calculates statistics about the content including character count,
    word count, sentence count, and estimated token count.
    
    Args:
        pdf_path: Path to the PDF file to process.
        page_offset: Number of pages to offset from the beginning 
                    (useful when PDF has cover pages). Default: 4
    
    Returns:
        A list of dictionaries, each containing:
            - page_number: Adjusted page number
            - page_char_count: Number of characters on page
            - page_word_count: Number of words on page
            - page_sentence_count_raw: Number of sentences (split by ". ")
            - page_token_count: Estimated token count (1 token ≈ 4 characters)
            - text: Extracted and formatted text
            
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If PDF cannot be opened or read.
    """
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty")
    
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise
    
    pages_and_texts = []
    
    for page_number, page in enumerate(doc):
        try:
            text = page.get_text()
            text = text_formatter(text)
            
            pages_and_texts.append({
                "page_number": page_number - page_offset,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token ≈ 4 characters
                "text": text
            })
        except Exception as e:
            logger.warning(f"Error processing page {page_number}: {e}")
            continue
    
    logger.info(f"Successfully extracted text from {len(pages_and_texts)} pages")
    return pages_and_texts


def add_sentences_to_pages(pages_and_texts: List[Dict[str, Any]]) -> None:
    """
    Split page text into sentences and add sentence count.
    
    Modifies pages_and_texts in-place by adding:
        - sentences: List of sentences
        - page_sentence_count_spacy: Number of sentences
    
    Args:
        pages_and_texts: List of page dictionaries from open_and_read_pdf().
    """
    for item in pages_and_texts:
        item["sentences"] = item["text"].split(". ")
        item["page_sentence_count_spacy"] = len(item["sentences"])
    
    logger.debug(f"Added sentence splitting to {len(pages_and_texts)} pages")


def explode_sentences_to_pages(pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert sentences to individual items with page number reference.
    
    Args:
        pages_and_texts: List of page dictionaries with sentences.
    
    Returns:
        List of dictionaries, each containing a single sentence and its page number.
    """
    pages_and_sentences = []
    
    for item in pages_and_texts:
        for sentence in item.get("sentences", []):
            pages_and_sentences.append({
                "page_number": item["page_number"],
                "sentence": sentence
            })
    
    logger.debug(f"Exploded into {len(pages_and_sentences)} individual sentences")
    return pages_and_sentences


def split_list(input_list: List[str], slice_size: int) -> List[List[str]]:
    """
    Split a list into sublists of specified size.
    
    Args:
        input_list: List to split.
        slice_size: Size of each sublist.
    
    Returns:
        List of sublists. The last sublist may be smaller than slice_size.
        
    Example:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def create_chunks_from_sentences(
    pages_and_texts: List[Dict[str, Any]],
    chunk_size: int
) -> List[Dict[str, Any]]:
    """
    Create text chunks by grouping sentences together.
    
    Args:
        pages_and_texts: List of page dictionaries with sentences.
        chunk_size: Number of sentences to group together.
    
    Returns:
        List of dictionaries, each containing:
            - page_number: Page number of the chunk
            - sentence_chunk: Joined sentences as a single string
            - chunk_char_count: Number of characters
            - chunk_word_count: Number of words
            - chunk_token_count: Estimated token count
    """
    # First add sentence-level chunks
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(
            input_list=item.get("sentences", []),
            slice_size=chunk_size
        )
        item["num_chunks"] = len(item["sentence_chunks"])
    
    # Then explode chunks to individual items
    pages_and_chunks = []
    
    for item in pages_and_texts:
        for sentence_chunk in item.get("sentence_chunks", []):
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": "".join(sentence_chunk).replace("  ", " ").strip(),
            }
            
            # Calculate chunk statistics
            chunk_dict["chunk_char_count"] = len(chunk_dict["sentence_chunk"])
            chunk_dict["chunk_word_count"] = len([
                word for word in chunk_dict["sentence_chunk"].split(" ")
            ])
            chunk_dict["chunk_token_count"] = len(chunk_dict["sentence_chunk"]) / 4
            
            pages_and_chunks.append(chunk_dict)
    
    logger.info(f"Created {len(pages_and_chunks)} chunks from {len(pages_and_texts)} pages")
    return pages_and_chunks
