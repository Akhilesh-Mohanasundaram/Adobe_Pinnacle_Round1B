import os
import subprocess
import sys

# --- Configuration ---
# Get the directory where this main.py script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output directories relative to BASE_DIR
PDF_DIR = os.path.join(BASE_DIR, "pdfs") # Your input PDFs go here
R1_OP_DIR = os.path.join(BASE_DIR, "r1_op") # p.py output goes here
OUTPUT_DIR = os.path.join(BASE_DIR, "output") # c1.py, c2.py, final_c3.py outputs go here

# Define paths for your existing scripts
P_SCRIPT = os.path.join(BASE_DIR, "p.py")
C1_SCRIPT = os.path.join(BASE_DIR, "c1.py")
C2_SCRIPT = os.path.join(BASE_DIR, "c2.py")
FINAL_C3_SCRIPT = os.path.join(BASE_DIR, "final_c3.py")

# --- Helper Function to Run Scripts ---
def run_script(script_path, script_name, *args):
    """
    Runs a Python script using subprocess and checks its exit code.
    Passes additional arguments to the script if provided.
    """
    if not os.path.exists(script_path):
        print(f"❌ Error: Script '{script_name}' not found at '{script_path}'. Please ensure it's in the correct directory.")
        sys.exit(1) # Exit if the script isn't found

    command = [sys.executable, script_path] + list(args)
    print(f"\n--- Running {script_name} ---")
    print(f"Executing: {' '.join(command)}")

    try:
        # Use run() for simpler execution and error handling
        # capture_output=False means output goes directly to console
        # check=True raises CalledProcessError if the command returns a non-zero exit code
        subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {script_name} failed with exit code {e.returncode}")
        print(f"Stderr: {e.stderr.decode()}")
        print(f"Stdout: {e.stdout.decode()}")
        sys.exit(1) # Exit if any script fails
    except FileNotFoundError:
        print(f"❌ Error: Python interpreter not found. Ensure Python is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)

# --- Main Execution Flow ---
if __name__ == "__main__":
    print(f"Starting the combined pipeline from: {BASE_DIR}")

    # 1. Create necessary directories if they don't exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(R1_OP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Checked/Created necessary directories: pdfs/, r1_op/, output/")

    # 2. Run p.py
    # IMPORTANT: Your p.py must be able to find and process PDFs in PDF_DIR
    # and save its output to R1_OP_DIR. You might need to modify p.py
    # to accept these directory paths as command-line arguments,
    # or ensure p.py inherently knows these relative paths.
    # For now, we assume p.py finds its inputs and outputs correctly relative
    # to its own location or current working directory.
    # If p.py needs specific arguments (e.g., a PDF filename), you'll add them here:
    # Example: run_script(P_SCRIPT, "p.py", "--pdf-dir", PDF_DIR, "--output-dir", R1_OP_DIR)
    print("\n##### Step 1: Running p.py (PDF Processing and Outline Extraction) #####")
    run_script(P_SCRIPT, "p.py") # Assuming p.py handles its own file system interactions

    # 3. Run c1.py
    # This script typically loads challenge1b_input.json (from BASE_DIR)
    # and the R1A outputs from R1_OP_DIR, saving to OUTPUT_DIR.
    print("\n##### Step 2: Running c1.py (Data Loading and Merging) #####")
    run_script(C1_SCRIPT, "c1.py")

    # 4. Run c2.py
    # This script typically loads data from OUTPUT_DIR (from c1.py),
    # extracts full text from PDFs (from PDF_DIR), saving updated data to OUTPUT_DIR.
    print("\n##### Step 3: Running c2.py (Full Text Extraction) #####")
    run_script(C2_SCRIPT, "c2.py")

    # 5. Run final_c3.py
    # This script typically loads data from OUTPUT_DIR (from c2.py),
    # performs embedding and similarity, and saves final output to OUTPUT_DIR.
    print("\n##### Step 4: Running final_c3.py (Section Embedding and Aggregation) #####")
    run_script(FINAL_C3_SCRIPT, "final_c3.py")

    print("\n🎉 All steps completed successfully through main.py!")