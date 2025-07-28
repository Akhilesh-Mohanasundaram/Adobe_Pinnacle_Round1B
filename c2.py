# step2.py

import json
import os
import fitz # PyMuPDF (Make sure you have PyMuPDF installed: pip install PyMuPDF)

# Define paths relative to the current working directory.
BASE_DIR = os.getcwd()

# Input/Output file paths for intermediate data
OUTPUT_DIR = os.path.join(BASE_DIR, "output") # Shared output directory
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure it exists

TEMP_STEP1_INPUT_FILE = os.path.join(OUTPUT_DIR, "temp_step1_data.json")
TEMP_STEP2_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "temp_step2_data.json")

# Directory where your actual PDF files are located
PDF_DIR = os.path.join(BASE_DIR, "pdfs")


def load_json_file(filepath):
    """Helper to load any JSON file, raising FileNotFoundError if not found."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_from_pdf(pdf_filepath):
    """
    Extracts text from each page of a PDF and returns it as a dictionary.
    Keys are page numbers (1-indexed), values are page text.
    Returns an empty dictionary if the file cannot be opened or processed.
    """
    page_texts = {}
    if not os.path.exists(pdf_filepath):
        print(f"  ERROR: PDF file not found at '{pdf_filepath}'. Cannot extract text.")
        return page_texts

    try:
        document = fitz.open(pdf_filepath)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text = page.get_text("text") # "text" extracts raw text
            page_texts[page_num + 1] = text # Use 1-indexed page numbers for consistency
        document.close()
        print(f"  Successfully extracted text from: {os.path.basename(pdf_filepath)}")
    except Exception as e:
        print(f"  ERROR during text extraction from {os.path.basename(pdf_filepath)}: {e}")
    return page_texts


if __name__ == "__main__":
    print(f"Starting Step 2 processing from: {BASE_DIR}")

    # --- Load data from Step 1's output ---
    try:
        step1_data = load_json_file(TEMP_STEP1_INPUT_FILE)
        challenge_data = step1_data["challenge_data"]
        loaded_r1a_outputs = step1_data["loaded_r1a_outputs"]
        print(f"Successfully loaded data from {TEMP_STEP1_INPUT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Step 1 data not found at '{TEMP_STEP1_INPUT_FILE}'. Please run step1.py first.")
        exit(1) # Exit with an error code
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{TEMP_STEP1_INPUT_FILE}'. File might be corrupted.")
        exit(1)

    print("\n--- Performing Full Text Extraction ---")
    # Iterate through the documents loaded from step1 and add their full text
    for doc_filename, doc_entry in loaded_r1a_outputs.items():
        full_pdf_filepath = os.path.join(PDF_DIR, doc_filename)
        doc_entry["full_text_pages"] = extract_text_from_pdf(full_pdf_filepath)

    print("\n--- Summary of Data After Step 2 ---")
    print("Documents with R1A outlines and text extracted:")
    for doc_filename, data in loaded_r1a_outputs.items():
        num_pages_extracted = len(data.get("full_text_pages", {}))
        print(f"- {doc_filename} (Pages: {num_pages_extracted}, Outline Entries: {len(data['outline'])})")

    # Example: Accessing text from a page to confirm
    if "South of France - Cities.pdf" in loaded_r1a_outputs:
        cities_pdf_data = loaded_r1a_outputs["South of France - Cities.pdf"]
        if 1 in cities_pdf_data.get("full_text_pages", {}):
            print("\n--- First 200 chars of Page 1 from 'South of France - Cities.pdf' ---")
            print(cities_pdf_data["full_text_pages"][1][:200] + "...")
        else:
            print("\nPage 1 text not available for 'South of France - Cities.pdf'.")

    # --- SAVE THE OUTPUT FOR THE NEXT STEP (step3.py) ---
    data_to_save_step2 = {
        "challenge_data": challenge_data, # Pass along the original challenge data
        "loaded_r1a_outputs_with_text": loaded_r1a_outputs # Now includes 'full_text_pages'
    }
    with open(TEMP_STEP2_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save_step2, f, indent=4) # Use indent for readability
    print(f"\nStep 2 data saved to: {TEMP_STEP2_OUTPUT_FILE}")

    print("\nStep 2 completed successfully.")
    print("Now, proceed to create 'step3.py' and run it.")