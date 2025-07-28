import json
import os

# Define paths relative to the current working directory.
BASE_DIR = os.getcwd()

# Input files
INPUT_FILE_PATH = os.path.join(BASE_DIR, "challenge1b_input.json")
R1_OP_DIR = os.path.join(BASE_DIR, "r1_op") # Directory for R1A JSON outputs

# Output directory for temporary data and final results
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# Define the path for the temporary file where step1.py will save its output
TEMP_STEP1_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "temp_step1_data.json")


def load_json_file(filepath):
    """Helper to load any JSON file, raising FileNotFoundError if not found."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_r1b_inputs():
    """
    Loads challenge1b_input.json and all corresponding Round 1A output JSONs
    by using the actual filenames provided in the input JSON.
    """
    challenge_data = load_json_file(INPUT_FILE_PATH)
    print("Loaded challenge1b_input.json")

    loaded_r1a_outputs_dict = {} # This will store all the outlines from R1A outputs

    # Iterate through the documents listed in the input JSON
    for doc_info in challenge_data.get("documents", []):
        # Get the actual PDF filename from the input JSON
        pdf_filename = doc_info.get("filename")
        if not pdf_filename:
            print("WARNING: Found a document entry in input JSON with no filename. Skipping.")
            continue

        # Derive the corresponding JSON filename from the PDF filename (e.g., "my_doc.pdf" -> "my_doc.json")
        json_filename_r1a = pdf_filename.replace(".pdf", ".json")
        r1a_output_path = os.path.join(R1_OP_DIR, json_filename_r1a)

        try:
            r1a_data = load_json_file(r1a_output_path)
            # Use the correct pdf_filename as the dictionary key
            loaded_r1a_outputs_dict[pdf_filename] = {
                "title": r1a_data.get("title", doc_info.get("title")),
                "outline": r1a_data.get("outline", []),
            }
            print(f"Loaded R1A output for PDF: '{pdf_filename}' from JSON: '{json_filename_r1a}'")
        except FileNotFoundError:
            print(f"WARNING: R1A output not found for PDF '{pdf_filename}' at expected path '{r1a_output_path}'. Skipping this document.")
            loaded_r1a_outputs_dict[pdf_filename] = {
                "title": doc_info.get("title"),
                "outline": [],
            }
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON for R1A output for PDF '{pdf_filename}' at '{r1a_output_path}'. Skipping.")
            loaded_r1a_outputs_dict[pdf_filename] = {
                "title": doc_info.get("title"),
                "outline": [],
            }

    return challenge_data, loaded_r1a_outputs_dict

if __name__ == "__main__":
    print(f"Starting Step 1 processing from: {BASE_DIR}")

    challenge_data, loaded_r1a_outputs = load_all_r1b_inputs()

    print("\n--- Summary of Loaded Data (Step 1) ---")
    print("Challenge ID:", challenge_data["challenge_info"]["challenge_id"])
    print("Persona:", challenge_data["persona"]["role"])
    print("Job to be Done:", challenge_data["job_to_be_done"]["task"])
    print("Documents with R1A outlines loaded:", list(loaded_r1a_outputs.keys()))

    # Example: Accessing outline for one document
    if loaded_r1a_outputs:
        first_pdf_key = list(loaded_r1a_outputs.keys())[0]
        print(f"\nOutline for {first_pdf_key} (first 3 entries):")
        for entry in loaded_r1a_outputs[first_pdf_key]["outline"][:3]:
            print(f"   Level: {entry['level']}, Text: {entry['text']}, Page: {entry['page']}")
    else:
        print("\nNo R1A outlines were loaded for any documents to show an example.")


    # --- SAVE THE OUTPUT FOR THE NEXT STEP (step2.py) ---
    data_to_save = {
        "challenge_data": challenge_data,
        "loaded_r1a_outputs": loaded_r1a_outputs
    }
    with open(TEMP_STEP1_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"\nStep 1 data saved to: {TEMP_STEP1_OUTPUT_FILE}")

    print("\nStep 1 completed successfully.")
    print("Now, proceed to run 'c2.py'.")

