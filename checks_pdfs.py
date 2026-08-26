import os
import shutil
from pypdf import PdfReader

SOURCE_FOLDER = "/Users/MSI/pdf-rag/pdf"   # e.g. "/Users/mohamed/Downloads/ensi-drive"
DEST_FOLDER = "data"

def has_text_layer(pdf_path, min_chars=50):
    try:
        reader = PdfReader(pdf_path)
        text = "".join((page.extract_text() or "") for page in reader.pages[:3])
        return len(text.strip()) > min_chars
    except Exception as e:
        print(f"  error reading {pdf_path}: {e}")
        return False

def main():
    kept, skipped = 0, 0
    for root, dirs, files in os.walk(SOURCE_FOLDER):   # walks into subfolders too
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, filename)
            if has_text_layer(path):
                shutil.copy(path, os.path.join(DEST_FOLDER, filename))
                print(f"✅ OCR'd (kept): {filename}")
                kept += 1
            else:
                print(f"❌ Handwritten/scanned (skipped): {filename}")
                skipped += 1
    print(f"\nDone. Kept {kept}, skipped {skipped}.")

if __name__ == "__main__":
    main()