import json
import os
from datetime import datetime
import re # Import for regular expressions
import time # Import the time module

# Import for Sentence-BERT
from sentence_transformers import SentenceTransformer
import torch

# Imports for later steps
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
output_dir = "output"
temp_step2_file = os.path.join(output_dir, "temp_step2_data.json")
intermediate_output_filename = os.path.join(output_dir, "temp_step3_2_data.json")
# Changed final output filename to match the expected challenge output naming convention
final_output_filename = os.path.join(output_dir, "challenge1b_output.json") 

# model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
# model_name = 'multi-qa-mpnet-base-dot-v1'
# model_name = 'intfloat/e5-small-v2'
# --- Define path to the local Sentence Transformer model ---
MODELS_DIR = "models"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, SBERT_MODEL_NAME)
# model_name='sentence-transformers/sentence-t5-base'

# Output limits
# Set to 7 as per your request for "top 7 matches only in the output json file"
NUM_TOP_SECTIONS_FOR_OUTPUT = 7 

# --- Helper Function for Robust Heading Matching (using regex) ---
def find_robust_heading_match(heading_text, page_content, start_offset=0):
    """
    Finds the start and end character index of a heading within page_content
    using regex to account for variable whitespace and case.

    Args:
        heading_text (str): The heading text to find.
        page_content (str): The full text of the page.
        start_offset (int): Character index in page_content to start searching from.

    Returns:
        tuple: (start_index, end_index) of the matched heading in page_content,
                or (None, None) if not found.
    """
    if not isinstance(heading_text, str) or not isinstance(page_content, str):
        return None, None

    # 1. Normalize internal spaces in heading_text to single spaces, then escape for regex
    normalized_heading_text = re.sub(r'\s+', ' ', heading_text).strip()
    escaped_normalized_heading = re.escape(normalized_heading_text)
    
    # 2. Replace the escaped space ('\ ') with a regex pattern for one or more whitespace chars ('\s+')
    heading_pattern = escaped_normalized_heading.replace(re.escape(' '), r'\s+')

    # Compile regex for case-insensitive and dotall (makes '.' match newlines too) search
    search_pattern = re.compile(heading_pattern, re.IGNORECASE | re.DOTALL)
    
    match = search_pattern.search(page_content, pos=start_offset)
    
    if match:
        return match.start(), match.end()
    else:
        return None, None

def clean_text(text):
    """
    Cleans text by removing common bullet points and normalizing whitespace.
    """
    # Remove common bullet points (•, *, -) at the start of lines
    text = re.sub(r'^[•*-]\s*', '', text, flags=re.MULTILINE)
    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Step 3.1 Code ---

try:
    with open(temp_step2_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Successfully loaded data from {temp_step2_file}")
except FileNotFoundError:
    print(f"❌ Error: {temp_step2_file} not found. Please ensure c2.py was run successfully.")
    exit()

challenge_data = data["challenge_data"]
loaded_docs_with_text = data["loaded_r1a_outputs_with_text"]

persona_description = challenge_data["persona"]["role"]
job_description = challenge_data["job_to_be_done"]["task"]

print(f"\n✨ Extracted Persona: '{persona_description}'")
print(f"✨ Extracted Job to be Done: '{job_description}'")

print(f"\n🧠 Loading local Sentence-BERT model from: '{SBERT_MODEL_PATH}'...")
try:
    # Point the SentenceTransformer to the local folder path
    embedding_model = SentenceTransformer(SBERT_MODEL_PATH)
    print("✅ Embedding model loaded successfully from local path.")
except Exception as e:
    print(f"❌ Error loading local embedding model from '{SBERT_MODEL_PATH}': {e}")
    print("💡 Please ensure the model exists and run 'downloader.py' if needed.")
    exit()

print("\n🚀 Generating embedding for the 'Job to be Done' description...")
job_embedding = embedding_model.encode(job_description, convert_to_tensor=True)
print(f"✅ Job embedding generated. Shape: {job_embedding.shape}")


# --- Step 3.2 Code (Revised for precise section content extraction with robust regex matching) ---

print("\n🔍 Preparing document sections for embedding using precise boundary detection and robust regex matching...")

document_sections = [] # List to hold data for each section

level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5, 'H6': 6} 

for doc_filename, doc_data in loaded_docs_with_text.items():
    outline = doc_data["outline"]
    full_text_pages = doc_data["full_text_pages"]
    
    last_page_in_doc = 0
    if full_text_pages: 
        last_page_in_doc = max([int(p) for p in full_text_pages.keys()])

    indexed_outline = []
    for i, heading in enumerate(outline):
        indexed_outline.append({
            "index": i,
            "text": heading["text"],
            "page": heading["page"],
            "level": heading["level"],
            "level_val": level_map.get(heading["level"], 99) 
        })
    
    indexed_outline.sort(key=lambda x: (x["page"], x["level_val"]))

    for i, current_heading in enumerate(indexed_outline):
        section_title = current_heading["text"]
        section_start_page = current_heading["page"]
        current_level_val = current_heading["level_val"]
        
        section_content_parts = []
        
        section_end_page_exclusive = last_page_in_doc + 1 
        next_section_start_title = None
        
        # Determine the page where the next higher or same-level heading starts (exclusive end for current section)
        for j in range(i + 1, len(indexed_outline)):
            next_heading = indexed_outline[j]
            if next_heading["level_val"] <= current_level_val:
                # If the next heading is on the same page, the section content ends on that page.
                # So the exclusive end page for the range should be current_heading["page"] + 1
                # to ensure 'current_heading["page"]' is included in the range.
                if next_heading["page"] == current_heading["page"]:
                    section_end_page_exclusive = current_heading["page"] + 1 
                else:
                    section_end_page_exclusive = next_heading["page"]
                next_section_start_title = next_heading["text"]
                break
            
        for p_num in range(section_start_page, min(section_end_page_exclusive, last_page_in_doc + 1)):
            page_content = full_text_pages.get(str(p_num), "")
            if not page_content:
                continue

            current_page_segment_start_idx = 0
            current_page_segment_end_idx = len(page_content)
            
            # --- Apply start-of-section cut on the starting page ---
            if p_num == section_start_page:
                title_match_start, title_match_end = find_robust_heading_match(section_title, page_content, start_offset=0)
                
                if title_match_start is not None:
                    current_page_segment_start_idx = title_match_end # Content starts AFTER the heading
                else:
                    pass # Keep silent if heading not found at start, take from page start
            
            # --- Apply end-of-section cut on the last relevant page ---
            if next_section_start_title and p_num == section_end_page_exclusive - 1:
                next_title_match_start, _ = find_robust_heading_match(next_section_start_title, page_content, start_offset=current_page_segment_start_idx)
                
                if next_title_match_start is not None:
                    current_page_segment_end_idx = next_title_match_start # Content ends BEFORE the next heading
                else:
                    pass # Keep silent if next heading not found, take rest of page

            extracted_segment = page_content[current_page_segment_start_idx : current_page_segment_end_idx].strip()
            
            if extracted_segment: 
                section_content_parts.append(extracted_segment)
            else:
                pass # Keep silent if no content extracted
        
        full_section_content = " ".join(section_content_parts).strip()
        
        # --- Apply text cleaning here ---
        cleaned_full_section_content = clean_text(full_section_content)

        section_content_for_embedding = f"{section_title}. {cleaned_full_section_content}"
        
        document_sections.append({
            "document": doc_filename,
            "page_number": section_start_page, 
            "section_title": section_title,
            "level": current_heading["level"],
            "section_content_for_embedding": section_content_for_embedding,
            "original_full_section_text": cleaned_full_section_content # Store the cleaned text
        })

print(f"\n📊 Prepared {len(document_sections)} document sections with precise content for embedding.")

section_contents_to_encode = [sec["section_content_for_embedding"] for sec in document_sections]
print(f"🚀 Generating embeddings for {len(section_contents_to_encode)} sections...")

# --- Start timing the embedding process ---
start_time_embedding = time.time()

section_embeddings = embedding_model.encode(section_contents_to_encode, convert_to_tensor=True)

# --- End timing and calculate duration ---
end_time_embedding = time.time()
processing_time_embedding_seconds = end_time_embedding - start_time_embedding

print(f"✅ Section embeddings generated. Shape: {section_embeddings.shape}")
print(f"⏱️ Embedding processing time: {processing_time_embedding_seconds:.4f} seconds.")


print("\n🧮 Calculating similarity scores for each section...")
similarities = cosine_similarity(job_embedding.unsqueeze(0), section_embeddings)[0]

for i, section in enumerate(document_sections):
    section["similarity_score"] = similarities[i].item()

print("✅ Similarity calculation complete. Each section now has a 'similarity_score'.")

# --- Save Intermediate Output for Verification ---
try:
    os.makedirs(output_dir, exist_ok=True)

    with open(intermediate_output_filename, 'w', encoding='utf-8') as f:
        json.dump(document_sections, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"❌ Error saving intermediate section data: {e}")

# --- Configuration for Step 3.3 ---
num_top_sections = NUM_TOP_SECTIONS_FOR_OUTPUT # Use the configured limit

# --- Step 3.3: Selection and Aggregation of Relevant Sections (Refined) ---
print("\nSelecting and aggregating the most relevant document sections (refined logic)... 🏆")

# Define generic section titles (case-insensitive) using regex for exact match
GENERIC_TITLES_REGEX = [
    r"^\s*introduction\s*$", r"^\s*conclusion\s*$", r"^\s*abstract\s*$", r"^\s*summary\s*$",
    r"^\s*table of contents\s*$", r"^\s*references\s*$", r"^\s*appendix(es)?\s*$", r"^\s*acknowledgements\s*$",
    r"^\s*foreword\s*$", r"^\s*preface\s*$", r"^\s*index\s*$", r"^\s*glossary\s*$", r"^\s*disclaimer\s*$",
    r"^\s*copyright\s*$", r"^\s*dedication\s*$", r"^\s*list of figures\s*$", r"^\s*list of tables\s*$",
    r"^\s*introduction\s+\d+\.?\s*$", # "Introduction 1", "Introduction 1.1" etc.
    r"^\s*conclusion\s+\d+\.?\s*$",   # "Conclusion 1", "Conclusion A" etc.
]
# Compile regex patterns for efficiency and case-insensitivity
COMPILED_GENERIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GENERIC_TITLES_REGEX]

def is_generic_section(section_title):
    """Checks if a section title is generic based on predefined patterns."""
    title_lower = section_title.lower().strip()
    for pattern in COMPILED_GENERIC_PATTERNS:
        if pattern.match(title_lower):
            return True
    return False

# Sort all sections by similarity score in descending order
sorted_sections = sorted(document_sections, key=lambda x: x["similarity_score"], reverse=True)

# Separate sections into informative and generic
informative_sections = []
generic_sections = []

for section in sorted_sections:
    # IMPORTANT: Also filter out empty sections at this early stage if they are truly content-less
    if not section["original_full_section_text"].strip():
        # print(f"Skipping empty section: '{section['section_title']}' from {section['document']} page {section['page_number']}")
        continue # Skip sections with no content

    if is_generic_section(section["section_title"]):
        generic_sections.append(section)
    else:
        informative_sections.append(section)

final_relevant_sections_pool = []
# Track unique canonical generic titles included (e.g., 'introduction', 'conclusion')
included_generic_canonical_titles = set() 

# Limit on specific generic types
MAX_ONE_INTRO = 1 
MAX_ONE_CONC = 1
current_intro_count = 0
current_conc_count = 0

# 1. Prioritize adding informative sections
for section in informative_sections:
    final_relevant_sections_pool.append(section)


# 2. Add generic sections with limits
for section in generic_sections:
    section_title_lower = section["section_title"].lower().strip()

    # Apply limits for specific generic types
    if "introduction" in section_title_lower and current_intro_count < MAX_ONE_INTRO:
        if "introduction" not in included_generic_canonical_titles: # Ensure only one intro is added
            final_relevant_sections_pool.append(section)
            current_intro_count += 1
            included_generic_canonical_titles.add("introduction")
    elif "conclusion" in section_title_lower and current_conc_count < MAX_ONE_CONC:
        if "conclusion" not in included_generic_canonical_titles: # Ensure only one conclusion is added
            final_relevant_sections_pool.append(section)
            current_conc_count += 1
            included_generic_canonical_titles.add("conclusion")
    else:
        # For other generic titles, add if its exact title (canonical) hasn't been added yet
        if section_title_lower not in included_generic_canonical_titles:
            final_relevant_sections_pool.append(section)
            included_generic_canonical_titles.add(section_title_lower)

# Sort the comprehensive pool by similarity score before filtering nested sections
final_relevant_sections_pool_sorted = sorted(final_relevant_sections_pool, key=lambda x: x["similarity_score"], reverse=True)


# --- Remove nested sections: keep only parent if both parent and child appear ---
def get_level_value(level):
    level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5, 'H6': 6}
    return level_map.get(level, 99)

# Sort by document, page, and heading level to correctly identify parent-child relationships for filtering
final_relevant_sections_pool_sorted.sort(key=lambda x: (x["document"], x["page_number"], get_level_value(x["level"])))

filtered_sections = []
seen_parents_for_filtering = [] # Keep track of parent sections already included

for sec in final_relevant_sections_pool_sorted:
    doc = sec["document"]
    page = sec["page_number"]
    level = get_level_value(sec["level"])

    is_nested_child = False
    for parent in seen_parents_for_filtering:
        parent_doc = parent["document"]
        parent_page = parent["page_number"]
        parent_level = get_level_value(parent["level"])

        # Check if the current section is a child of a previously added parent section
        if (parent_doc == doc and
            parent_page <= page and # Child section can be on the same page or later
            parent_level < level):
            is_nested_child = True
            break

    if not is_nested_child:
        filtered_sections.append(sec)
        seen_parents_for_filtering.append(sec)

print(f"\n🧹 Initial filter for nested child sections. Count after filter: {len(filtered_sections)}")


# --- Final selection of relevant sections (considering empty content and backfill) ---
relevant_sections_final = []
seen_sections_keys_for_final = set() # To prevent duplicates

# Add sections from filtered_sections, ensuring they are not duplicates
for sec in filtered_sections:
    section_key = (sec["document"], sec["page_number"], sec["section_title"])
    if section_key not in seen_sections_keys_for_final:
        relevant_sections_final.append(sec)
        seen_sections_keys_for_final.add(section_key)

# If we still don't have enough sections, backfill from the original sorted_sections
# (which includes all sections, sorted by similarity, and already filtered for emptiness)
if len(relevant_sections_final) < NUM_TOP_SECTIONS_FOR_OUTPUT:
    # Iterate through ALL sections sorted by similarity (this includes informative and generic)
    for section in sorted_sections:
        section_key = (section["document"], section["page_number"], section["section_title"])
        
        # Ensure section is not already selected and has content
        if section_key not in seen_sections_keys_for_final and section["original_full_section_text"].strip():
            # Re-check for nesting against the *already selected* relevant_sections_final
            is_nested_child_during_backfill = False
            for selected_sec in relevant_sections_final:
                if (selected_sec["document"] == section["document"] and
                    selected_sec["page_number"] <= section["page_number"] and
                    get_level_value(selected_sec["level"]) < get_level_value(section["level"])):
                    is_nested_child_during_backfill = True
                    break
            
            # Re-check generic limits for backfilled sections
            is_allowed_generic = True
            if is_generic_section(section["section_title"]):
                section_title_lower = section["section_title"].lower().strip()
                if "introduction" in section_title_lower and "introduction" in included_generic_canonical_titles:
                    is_allowed_generic = False
                elif "conclusion" in section_title_lower and "conclusion" in included_generic_canonical_titles:
                    is_allowed_generic = False
                elif section_title_lower in included_generic_canonical_titles:
                    is_allowed_generic = False
                
            if not is_nested_child_during_backfill and is_allowed_generic:
                relevant_sections_final.append(section)
                seen_sections_keys_for_final.add(section_key)
                
                # Update generic counts/sets for backfilled generic sections
                if is_generic_section(section["section_title"]):
                    section_title_lower = section["section_title"].lower().strip()
                    if "introduction" in section_title_lower:
                        current_intro_count += 1
                        included_generic_canonical_titles.add("introduction")
                    elif "conclusion" in section_title_lower:
                        current_conc_count += 1
                        included_generic_canonical_titles.add("conclusion")
                    else:
                        included_generic_canonical_titles.add(section_title_lower)

                if len(relevant_sections_final) >= NUM_TOP_SECTIONS_FOR_OUTPUT:
                    break

# Finally, sort by similarity score and trim to the exact required number
relevant_sections_final = sorted(relevant_sections_final, key=lambda x: x["similarity_score"], reverse=True)[:NUM_TOP_SECTIONS_FOR_OUTPUT]

print(f"✅ Selected {len(relevant_sections_final)} relevant sections after content filtering and backfill.")
if not relevant_sections_final:
    print("❌ No relevant sections with content found to aggregate.")


# Aggregate the content of the selected sections (for internal use or if needed for other parts)
aggregated_context = ""
if relevant_sections_final:
    relevant_sections_sorted_for_aggregation = sorted(relevant_sections_final, key=lambda x: (x["document"], x["page_number"]))
    
    for section in relevant_sections_sorted_for_aggregation:
        aggregated_context += f"### {section['section_title']}\n{section['original_full_section_text']}\n\n"
    
    aggregated_context = aggregated_context.strip()
else:
    aggregated_context = "No relevant information found in the provided documents." # Fallback text


# --- Final Output Construction (matching challenge1b_output.json format) ---

# 1. Prepare metadata
input_documents_list = [doc["filename"] for doc in challenge_data["documents"]]
processing_timestamp = datetime.now().isoformat()

metadata_output = {
    "input_documents": input_documents_list,
    "persona": persona_description,
    "job_to_be_done": job_description, # This already holds just the task
    "processing_timestamp": processing_timestamp
}

# 2. Prepare extracted_sections
extracted_sections_output = []
# Sort by similarity_score in descending order to assign importance_rank
ranked_sections_for_output = sorted(relevant_sections_final, key=lambda x: x["similarity_score"], reverse=True)

for i, section in enumerate(ranked_sections_for_output):
    extracted_sections_output.append({
        "document": section["document"],
        "section_title": section["section_title"],
        "importance_rank": i + 1, # 1-indexed rank
        "page_number": section["page_number"]
    })

# 3. Prepare subsection_analysis
subsection_analysis_output = []
# Use the same NUM_TOP_SECTIONS_FOR_OUTPUT for consistency, or adjust if a different number is desired here
num_subsections_to_process = min(NUM_TOP_SECTIONS_FOR_OUTPUT, len(ranked_sections_for_output)) 

for i in range(num_subsections_to_process):
    section = ranked_sections_for_output[i]
    # 'original_full_section_text' now contains the cleaned text
    refined_text_content = section["original_full_section_text"].strip() 
    
    # Only add to output if the refined text content is not empty
    if refined_text_content:
        subsection_analysis_output.append({
            "document": section["document"],
            "refined_text": refined_text_content,
            "page_number": section["page_number"]
        })

# Construct the final JSON object
final_output_challenge1b = {
    "metadata": metadata_output,
    "extracted_sections": extracted_sections_output,
    "subsection_analysis": subsection_analysis_output
}

try:
    os.makedirs(output_dir, exist_ok=True)
    with open(final_output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output_challenge1b, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 Round 1B final output complete! Saved to '{final_output_filename}'")
except Exception as e:
    print(f"❌ Error saving final Round 1B output: {e}")

# 'document_sections' is still in memory and ready for Step 4.