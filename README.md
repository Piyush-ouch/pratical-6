# Retrieval-Augmented Generation (RAG) using a Pre-trained LLM

A practical implementation of a **Retrieval-Augmented Generation (RAG)**
application that answers questions from a domain-specific document
collection. The project compares responses from a pre-trained Large
Language Model (LLM) with and without retrieved document context.

## Project Overview

A standard LLM may not know private or domain-specific information that
was not included in its training data. RAG improves this by retrieving
relevant passages from an external knowledge base and adding them to the
prompt before the model generates an answer.

This project uses a fictional **Northbridge Technical University (NTU)**
handbook as its knowledge base. It contains information about academic
rules, library and laboratory policies, and hostel and student welfare
guidelines.

## Objectives

-   Load and prepare domain-specific text documents.
-   Clean and split documents into overlapping text chunks.
-   Generate semantic embeddings for each chunk.
-   Store and search embeddings using ChromaDB.
-   Retrieve the most relevant passages for a user query.
-   Generate answers using a pre-trained LLM.
-   Compare LLM-only responses with RAG responses.
-   Evaluate retrieval success and answer correctness.
-   Test how the RAG system handles questions that are outside its
    document collection.

## Technology Stack

  ------------------------------------------------------------------------------
  Component                           Technology
  ----------------------------------- ------------------------------------------
  Language                            Python

  Notebook environment                Google Colab

  Embedding model                     `sentence-transformers/all-MiniLM-L6-v2`

  Vector database                     ChromaDB

  Pre-trained LLM                     `Qwen/Qwen2.5-1.5B-Instruct`

  Model and NLP libraries             Transformers, PyTorch,
                                      Sentence-Transformers

  Evaluation and tabular output       pandas
  ------------------------------------------------------------------------------

## RAG Workflow

``` text
Domain-specific documents
          |
          v
   Text loading & cleaning
          |
          v
     Text chunking
          |
          v
   Embedding generation
          |
          v
    ChromaDB vector store
          |
          v
       User query
          |
          v
    Query embedding
          |
          v
  Similarity-based retrieval
          |
          v
  Relevant context + query
          |
          v
  Pre-trained language model
          |
          v
   Context-grounded answer
```

## Knowledge Base

The notebook creates three fictional university handbook documents:

  -----------------------------------------------------------------------
  Document                            Contents
  ----------------------------------- -----------------------------------
  `academic_rules.txt`                Attendance eligibility, backlog
                                      limits, examination rules, grading
                                      and re-evaluation

  `library_and_labs.txt`              Library timings, book and laptop
                                      borrowing, late fees and computer
                                      laboratory rules

  `hostel_and_welfare.txt`            Hostel gate timings, visitor
                                      policy, mess schedule, grievance
                                      support and anti-ragging rules
  -----------------------------------------------------------------------

The dataset contains sample facts such as the minimum attendance
requirement, library late fee, hostel gate closing time and laptop
borrowing period. These facts are used as expected answers during
evaluation.

## Implementation Details

### 1. Document loading and cleaning

The handbook text files are loaded into Python. The cleaning process
removes carriage returns, normalizes repeated spaces and tabs, and
reduces excessive blank lines.

### 2. Text chunking

Documents are split into overlapping text segments.

-   Chunk size: **400 characters**
-   Chunk overlap: **80 characters**

Overlapping chunks help preserve context when a relevant fact is close
to a chunk boundary.

### 3. Embedding generation

The `all-MiniLM-L6-v2` sentence-transformer converts each text chunk
into a 384-dimensional vector representation. The same embedding model
is used to encode incoming user queries.

### 4. Vector database and retrieval

ChromaDB stores the embeddings, original text and source metadata.
Cosine distance is used to retrieve the most semantically similar
chunks. The notebook uses the top three retrieved chunks by default.

### 5. LLM response generation

The project uses `Qwen/Qwen2.5-1.5B-Instruct` in two modes:

-   **LLM-only:** The model receives the question without handbook
    context.
-   **RAG:** The model receives the question and the retrieved passages.
    It is instructed to answer only from the supplied context, cite the
    source number, and indicate when the documents do not contain the
    answer.

## Run the Project

### Option 1: Google Colab (Recommended)

1.  Open the notebook using the link below.
2.  If required, select a runtime with GPU acceleration.
3.  Run the installation cell.
4.  Execute the notebook cells in order.
5.  Wait for the embedding index and language model to load.
6.  Review the retrieved passages, generated answers and evaluation
    table.

**Google Colab Notebook:**\
[Open Practical No. 06 -- RAG
Notebook](https://colab.research.google.com/drive/1v3q-wjBoWxYYnrTXlZ3wl87wEj0toLNV?usp=sharing)

### Option 2: Run Locally

Clone or download the repository, then install the dependencies:

``` bash
pip install -U transformers accelerate sentence-transformers chromadb pypdf pandas torch
```

Open the notebook in Jupyter Notebook or JupyterLab and run the cells
sequentially. A compatible GPU is recommended for faster LLM inference.
CPU execution may be slower.

## Evaluation

The notebook tests questions about:

1.  Minimum attendance required for examinations.
2.  Library late fee per day.
3.  Hostel gate closing time.
4.  Backlogs allowed for promotion.
5.  Head of the Student Grievance Cell.
6.  Examination form late fee.
7.  Laptop borrowing duration.
8.  Re-evaluation fee.
9.  Capital of France as a general-knowledge control question.

The evaluation table includes the expected fact, LLM-only answer, RAG
answer, retrieval hit, and simple fact-presence checks. The notebook
exports the generated results to:

``` text
rag_vs_llm_results.csv
```

The exact model responses and evaluation scores are generated when the
notebook is run. They may vary depending on the model version and
runtime environment.

## Out-of-Scope Test

The notebook also asks:

> What is the tuition fee for the MBA programme at NTU?

This information is not present in the handbook. The RAG prompt asks the
model to respond that it does not know based on the provided documents,
rather than inventing a value.

## Expected Learning Outcomes

After completing this practical, the learner should be able to:

-   Explain the purpose and architecture of a RAG pipeline.
-   Prepare documents for semantic retrieval.
-   Generate and use text embeddings.
-   Create a vector collection and perform similarity search.
-   Connect retrieved context to a pre-trained LLM.
-   Compare retrieval-grounded generation with LLM-only generation.
-   Identify retrieval limitations and unsupported model responses.

## Limitations and Future Improvements

-   Fixed chunk sizes may split related information or combine unrelated
    content.
-   A fixed top-k value may fail to retrieve the best passage.
-   A compact LLM may not always follow context restrictions correctly.
-   Simple keyword matching is not a complete measure of factual
    faithfulness.

Possible improvements include hybrid keyword and vector search (BM25 +
embeddings), cross-encoder re-ranking, metadata filtering, improved
chunking strategies, larger instruction-tuned models and human
evaluation of answer faithfulness.

## Conclusion

This practical demonstrates a complete RAG workflow using document
cleaning, overlapping chunking, semantic embeddings, ChromaDB retrieval
and a pre-trained instruction-following LLM. By supplying relevant
source passages during generation, RAG can make domain-specific answers
more grounded and traceable than responses generated without retrieval.
The quality of the result depends on retrieving the correct passages and
on the language model using the supplied context appropriately.

------------------------------------------------------------------------

**Practical:** No. 06 -- Retrieval-Augmented Generation (RAG) using a
Pre-trained LLM\
**Student:** Piyush Jangade
