import os
import glob
from pypdf import PdfReader

def extract_text_from_pdfs(pdf_dir: str) -> list:
    """Extracts text from all PDF files in a directory."""
    documents = []
    print(f"🚀 Starting PDF extraction from {pdf_dir}...")
    
    # Use glob to find all PDF files recursively
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"), recursive=True)
    
    for file_path in pdf_files:
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Store as a simple dictionary
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1 
                        }
                    })
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            continue
            
    print(f"✅ Extracted text from {len(pdf_files)} files into {len(documents)} pages.")
    return documents
