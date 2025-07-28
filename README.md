# 📄 Persona-Based PDF Content Analysis Pipeline

## 1. Overview

This project provides a robust, containerized pipeline for deep analysis of PDF documents. It ingests a collection of PDFs and a user-defined persona, then intelligently extracts and ranks the most relevant sections based on the specified task. The entire workflow is packaged within a Docker container, ensuring consistent, dependency-free execution across any environment.

### Key Features

* ✅ **Automated Document Outlining:** Extracts hierarchical outlines (Title, H1, H2, etc.) from PDFs using machine learning models.
* ✅ **Semantic Section Ranking:** Uses Sentence-BERT embeddings to calculate the relevance of each document section to a given "job to be done."
* ✅ **Persona-Driven Analysis:** The relevance ranking is tailored to the specified user persona and their task.
* ✅ **Offline First:** All required ML models (spaCy, Sentence-BERT) are downloaded and stored locally, allowing the container to run without an internet connection.
* ✅ **Flexible and Scalable:** Processes any number of PDFs with arbitrary filenames.

---

## 2. 📁 Project Structure

For the pipeline to work correctly, your project should be organized as follows:

```
.
├── test
│   ├── pdfs/
│   │   ├── document1.pdf
│   │   └── another_document.pdf
│   └── challenge1b_input.json
│   └── challenge1b_output.json
├── models
├── c1.py
├── c2.py
├── final_c3.py
├── main_workflow.py
├── p.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 3. Prerequisites

### Step 3.1: Install Prerequisites
Ensure you have the following software installed on your system:
* **Git:** For cloning the project repository.
* **Python 3.9+:** Required for running local setup scripts.
* **Docker:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running.

### Step 3.2: Clone the Repository
Clone this project to your local machine using Git, then navigate into the newly created project directory.
```bash
git clone https://github.com/Jothika1526/Adobe_Pinnacle_Round1B
cd solution_round1b

---

## 4. One-Time Setup

Before running the pipeline for the first time, you need to download the necessary machine learning models and build the Docker image.

### Step 4.1: Build the Docker Image

Build the Docker image using the `docker build` command. We'll tag it as `my-app`. This process packages all scripts, models, and dependencies into a self-contained image.

```bash
docker build -t my-app .
```

---

## 5. Running the Pipeline with Your Test Case

Follow these steps to process the set of documents located in the `test` directory.

### Step 5.1: Execute the Docker Container

Run the pipeline using the `docker run` command. This command uses **volume mounts** (`-v`) to link the files from your local `test` directory to the directories inside the container.

Run the command below, replacing `"YOUR_FULL_PATH_TO_PROJECT"` with the absolute path you just copied.

#### **For Windows (PowerShell):**

```powershell
# Replace "YOUR_FULL_PATH_TO_PROJECT" with your actual path, e.g., "D:\Users\You\Documents\FINAL2"
docker run --rm -v "YOUR_FULL_PATH_TO_PROJECT\test:/app/output" my-app
```

#### **For macOS / Linux (Bash):**

```bash
# Replace "YOUR_FULL_PATH_TO_PROJECT" with your actual path, e.g., "/home/user/documents/FINAL2"
docker run --rm -v "YOUR_FULL_PATH_TO_PROJECT/test":/app/output my-app
```

The final output, `challenge1b_output.json`, will be saved back into your `test` folder upon completion.

Also, you can execute with other test cases by following below steps.

#### **For Windows (PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}/test/pdfs:/app/pdfs" `
  -v "${PWD}/test/challenge1b_input.json:/app/challenge1b_input.json" `
  -v "${PWD}/test:/app/output" `
  my-app
```

#### **For macOS / Linux (Bash):**

```bash
docker run --rm \
  -v "$(pwd)/test/pdfs":/app/pdfs \
  -v "$(pwd)/test/challenge1b_input.json":/app/challenge1b_input.json \
  -v "$(pwd)/test":/app/output \
  my-app
```

### How the Command Works

* `-v ".../test/pdfs:/app/pdfs"`: Maps your local `test/pdfs` folder to the input directory inside the container.
* `-v ".../test/challenge1b_input.json:/app/challenge1b_input.json"`: Maps your local input JSON to the file the container expects to read.
* `-v ".../test:/app/output"`: Maps your local `test` folder to the output directory inside the container. **This is how the final output JSON is saved back to your machine.**
* `my-app`: The name of the Docker image you built.

---

## 6. Pipeline Workflow

The `main_workflow.py` script orchestrates the execution of the pipeline in the following sequence:

1.  **`p.py`**: Extracts a structural outline from each PDF and saves it as a JSON file.
2.  **`c1.py`**: Reads the main `challenge1b_input.json` and merges it with the outlines generated in the previous step.
3.  **`c2.py`**: Extracts the full plain text from each page of the PDFs.
4.  **`final_c3.py`**:
    * Generates embeddings for the "job to be done" and each document section.
    * Calculates cosine similarity to find the most relevant sections.
    * Constructs and saves the final `challenge1b_output.json`.
