import fitz  # PyMuPDF
from typing import Dict, Any

class PDFParser:
    """Parser for extracting text from PDF files using PyMuPDF"""

    def __init__(self):
        pass

    def extract_text_from_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from PDF file bytes using PyMuPDF

        Args:
            file_content: PDF file content as bytes
            filename: Name of the PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=file_content, filetype="pdf")

            # Extract metadata
            metadata = {
                "filename": filename,
                "num_pages": doc.page_count,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            }

            # Extract text from all pages
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()

                if page_text and page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            doc.close()

            # Combine all text
            full_text = "\n\n".join(text_parts)

            # Clean text
            full_text = self._clean_text(full_text)

            if not full_text or not full_text.strip():
                return {
                    "text": "",
                    "metadata": metadata,
                    "success": False,
                    "error": "No text content found in PDF. This may be a scanned image PDF."
                }

            return {
                "text": full_text,
                "metadata": metadata,
                "success": True
            }

        except Exception as e:
            return {
                "text": "",
                "metadata": {"filename": filename},
                "success": False,
                "error": str(e)
            }

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(cleaned_lines)

        # Remove excessive blank lines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        return text

# Global parser instance
pdf_parser = PDFParser()
