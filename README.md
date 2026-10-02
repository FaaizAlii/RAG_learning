# RAG Learning --- Document Ingestion

This repository is a hands-on learning project for understanding
**Retrieval-Augmented Generation (RAG)** step by step.

The goal is to build the RAG pipeline incrementally rather than jumping
directly into a complete implementation.

Current focus:

> **SQLite data → Pydantic models → JSON → LangChain `JSONLoader` →
> LangChain `Document`**

The next steps will be added to this README as they are learned.

------------------------------------------------------------------------

## Project Data

The example project is a small shop database stored in SQLite:

``` text
shop.db
```

The database contains:

-   `customers`
-   `products`
-   `orders`
-   `order_items`

The database was seeded with Pakistani customer data and products such
as keyboards, monitors, mice, routers, SSDs, etc.

------------------------------------------------------------------------

# 1. RAG Pipeline

The overall pipeline we are learning is:

``` text
SQLite
   ↓
Pydantic
   ↓
JSON
   ↓
JSONLoader
   ↓
LangChain Documents
   ↓
Document Splitting
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retriever
   ↓
LLM + Retrieved Context
```

The important part is that we are learning each stage separately.

------------------------------------------------------------------------

# 2. Creating Structured Documents

The first step was to transform the relational SQLite data into a
structure that represents a customer and their orders.

We use Pydantic for this.

## Pydantic schema

``` python
from typing import List
from pydantic import BaseModel


class PurchasedProduct(BaseModel):
    product_name: str
    quantity: int
    price: float


class CustomerOrder(BaseModel):
    order_id: int
    status: str
    total: float
    products: List[PurchasedProduct]


class CustomerDocument(BaseModel):
    customer_id: int
    name: str
    email: str
    city: str
    orders: List[CustomerOrder]
```

The structure is:

``` text
CustomerDocument
│
├── customer_id
├── name
├── email
├── city
└── orders
    │
    ├── order_id
    ├── status
    ├── total
    └── products
        │
        ├── product_name
        ├── quantity
        └── price
```

This allows the relational database information to be represented as a
single structured customer document.

------------------------------------------------------------------------

# 3. Creating the JSON File

The Pydantic objects are converted to dictionaries and then written to
JSON.

With Pydantic v2:

``` python
import json

data = [
    document.model_dump()
    for document in documents
]

with open("customers.json", "w") as f:
    json.dump(data, f, indent=2)
```

The resulting file looks conceptually like:

``` json
[
  {
    "customer_id": 1,
    "name": "Sadaf Iqbal",
    "email": "sadafiqbal@gmail.com",
    "city": "Kabirwala",
    "orders": [
      {
        "order_id": 1,
        "status": "completed",
        "total": 661,
        "products": [
          {
            "product_name": "Mechanical Keyboard",
            "quantity": 3,
            "price": 120
          }
        ]
      }
    ]
  }
]
```

At this point we have moved from:

``` text
SQLite relational data
```

to:

``` text
structured JSON data
```

------------------------------------------------------------------------

# 4. Loading JSON with LangChain

The next step is to load the JSON file using LangChain's `JSONLoader`.

Install the community package if necessary:

``` bash
pip install langchain-community
```

Import the loader:

``` python
from langchain_community.document_loaders import JSONLoader
```

The basic loader is:

``` python
loader = JSONLoader(
    file_path="customers.json",
    jq_schema=".[]",
    text_content=False,
)

docs = loader.load()
```

## What does `.[]` mean?

Our JSON file contains an array:

``` json
[
    {...},
    {...},
    {...}
]
```

The jq expression:

``` text
.[] 
```

means:

> Take every object inside the JSON array.

Therefore, each customer becomes one LangChain `Document`.

For example:

``` text
customers.json
│
├── Customer 1
├── Customer 2
├── Customer 3
└── ...
```

becomes:

``` text
Document 1 → Customer 1
Document 2 → Customer 2
Document 3 → Customer 3
...
```

------------------------------------------------------------------------

# 5. Understanding LangChain `Document`

After loading:

``` python
for doc in docs[:2]:
    print(doc)
    print("=" * 80)
```

we get something similar to:

``` text
page_content='{"customer_id": 1, "name": "Sadaf Iqbal", ...}'
metadata={
    'source': '/path/to/customers.json',
    'seq_num': 1
}
```

A LangChain `Document` can be thought of as:

``` text
Document
│
├── page_content
│
└── metadata
```

## `page_content`

This contains the actual information that we eventually want to
retrieve.

For example:

``` text
customer information
orders
products
quantities
prices
statuses
```

## `metadata`

This contains information **about the document**.

Initially, `JSONLoader` gives us metadata such as:

``` python
{
    "source": ".../customers.json",
    "seq_num": 1
}
```

------------------------------------------------------------------------

# 6. Improving Metadata with `metadata_func`

The default metadata is not enough for our use case.

We want a small amount of useful information that identifies or
describes the document.

We use `metadata_func` for this.

``` python
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["customer_id"] = record["customer_id"]
    metadata["customer_name"] = record["name"]
    metadata["city"] = record["city"]
    metadata["email"] = record["email"]

    metadata["order_count"] = len(record["orders"])

    return metadata
```

The `record` argument represents the current JSON object.

For example:

``` python
{
    "customer_id": 1,
    "name": "Sadaf Iqbal",
    "email": "sadafiqbal@gmail.com",
    "city": "Kabirwala",
    "orders": [...]
}
```

The `metadata` argument contains the metadata that `JSONLoader` has
already created.

We add our own fields to it and return it.

------------------------------------------------------------------------

# 7. Using `metadata_func` with JSONLoader

The loader becomes:

``` python
loader = JSONLoader(
    file_path="customers.json",
    jq_schema=".[]",
    content_key=None,
    metadata_func=metadata_func,
)

docs = loader.load()
```

We can inspect the result:

``` python
for doc in docs[:2]:
    print("PAGE CONTENT:")
    print(doc.page_content)

    print("\nMETADATA:")
    print(doc.metadata)

    print("=" * 80)
```

The metadata should now look approximately like:

``` python
{
    "source": "/path/to/customers.json",
    "seq_num": 1,
    "customer_id": 1,
    "customer_name": "Sadaf Iqbal",
    "city": "Kabirwala",
    "email": "sadafiqbal@gmail.com",
    "order_count": 4
}
```

------------------------------------------------------------------------

# 8. Why Metadata?

The important distinction is:

``` text
page_content
    ↓
The actual information contained in the document

metadata
    ↓
Small attributes describing or identifying the document
```

For this project, the metadata is intentionally kept small:

``` text
customer_id
customer_name
city
email
order_count
```

We do **not** want to duplicate the complete customer/order information
inside metadata.

The detailed information already belongs in `page_content`.

A useful mental model is:

``` text
Document
│
├── page_content
│   └── "What does this customer have?"
│
└── metadata
    └── "Which customer/document is this?"
```

`customer_id` is the unique identifier for the customer, while fields
such as `city` can later be useful for filtering.

For example, conceptually, a vector store could eventually use metadata
to restrict retrieval to:

``` text
city = "Kabirwala"
```

That is different from semantic similarity.

------------------------------------------------------------------------

# 9. Current Notebook Progress

The notebook is currently organized into separate cells so that each
stage can be understood independently.

Current progress:

``` text
Cell 1
Import JSONLoader
        ↓
Cell 2
Define metadata_func
        ↓
Cell 3
Create JSONLoader
        ↓
Cell 4
Load documents
        ↓
Cell 5
Inspect page_content and metadata
```

The important thing at this stage is understanding what `JSONLoader`
actually produces.

We currently have:

``` text
SQLite
   ↓
Pydantic
   ↓
customers.json
   ↓
JSONLoader
   ↓
LangChain Documents
```

------------------------------------------------------------------------
# 10. Current Understanding

The main concepts learned so far:

### Pydantic

Used to define predictable structures for our data before creating JSON
documents.

### JSON

Acts as an intermediate document format between our relational database
and LangChain.

### JSONLoader

Reads the JSON file and converts individual JSON records into LangChain
`Document` objects.

### `jq_schema`

Controls which parts of the JSON are turned into documents.

For our array of customers:

```python
jq_schema=".[]"
```

means each customer becomes one document.

### `page_content`

Contains the actual document content.

### `metadata`

Contains small attributes describing the document.

### `metadata_func`

Allows us to customize the metadata generated by `JSONLoader`.

---

## Document Chunking

Documents often need to be split into smaller pieces before creating
embeddings.

For structured data such as our customer/order JSON, meaningful boundaries
are more useful than blindly splitting text.

For example:

```text
Customer
    ↓
Orders
    ↓
One order = one meaningful chunk
```

For normal documents such as PDFs, we can use a text splitter such as
`RecursiveCharacterTextSplitter`.

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)
```

### `chunk_size`

Controls approximately how large each chunk can become.

### `chunk_overlap`

Keeps some text from the previous chunk in the next chunk so that
information near chunk boundaries is not completely separated.

An important RAG practice is to inspect the generated chunks before
embedding them.

---

## Embeddings

An embedding converts text into a numerical vector representing its
semantic meaning.

We used:

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

Example:

```python
embedding = embedding_model.encode(
    "What infection prevention measures are recommended?"
)
```

The result is a numerical vector that can be compared with vectors from
other pieces of text.

The same embedding model must be used for both:

```text
Document
    ↓
Embedding

Query
    ↓
Embedding
```

This allows us to compare the semantic similarity between the query and
stored documents.

---

## Vector Store / ChromaDB

We used ChromaDB as our vector database.

```python
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="who_ipc"
)
```

Each stored item contains:

```text
ID
Document text
Embedding
Metadata
```

For example:

```python
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)
```

ChromaDB can then search for documents whose embeddings are semantically
similar to a query embedding.

---

## Retrieval

Retrieval means finding the most relevant chunks from the vector store
for a user's question.

Example:

```python
query = "What infection prevention measures are recommended?"

query_embedding = embedding_model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)
```

The retrieved documents become the context that can later be provided to
an LLM.

The basic flow is:

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB
      ↓
Most Relevant Chunks
```

### Important observation

Vector search is based on semantic similarity, not exact database lookups.

For our structured customer data, this meant that a question such as:

```text
What did Sadaf order?
```

could retrieve other customers before Sadaf's records.

Metadata filtering can therefore be useful when we have known structured
attributes:

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={
        "customer_name": "Sadaf Iqbal"
    },
)
```

This demonstrated an important distinction:

```text
Vector similarity
        +
Metadata filtering
```

can be combined when appropriate.

---

# RAG vs Traditional Database Tools

We also learned that a vector database is not automatically a replacement
for our normal database or database tools.

For structured information, traditional database queries are usually more
appropriate.

For example:

```text
"How much has Sadaf spent?"
        ↓
Database tool
```

```text
"How many completed orders does Sadaf have?"
        ↓
Database tool
```

For unstructured or semi-structured information:

```text
"What does our return policy say about monitors?"
        ↓
RAG
```

The two approaches can therefore coexist:

```text
                         User
                           ↓
                          LLM
                    ┌──────┴──────┐
                    ↓             ↓
             Database Tools      RAG
                    ↓             ↓
                 SQL DB       Vector DB
                    └──────┬──────┘
                           ↓
                          LLM
                           ↓
                         Answer
```

---

# PDF RAG

We then started a separate RAG exercise using a small WHO PDF about
infection prevention and control.

The PDF is loaded using `PyPDFLoader`:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdfs/who_ipc.pdf")

documents = loader.load()
```

Each PDF page becomes a LangChain `Document`.

For example:

```text
PDF
 ↓
Page 1 → Document
Page 2 → Document
Page 3 → Document
Page 4 → Document
```

We then split the PDF documents into smaller chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)
```

Unlike our structured customer data, the PDF does not have natural
database entities such as customers and orders, so a text splitter is
appropriate.

The PDF metadata, such as the source and page number, is preserved when
the documents are split.

---

# Retrieval + LLM

After testing retrieval independently, we connected the retrieved
documents to an LLM.

The overall process became:

```text
User Question
      ↓
Embedding Model
      ↓
ChromaDB
      ↓
Relevant Chunks
      ↓
Context
      ↓
LLM
      ↓
Final Answer
```

For example:

```python
docs = retrieve(
    "What infection prevention measures are recommended?"
)

context = "\n\n".join(docs)

prompt = f"""
Answer the question using ONLY the information provided in the context.

Context:
{context}

Question:
What infection prevention measures are recommended?

If the answer is not present in the context, say:
"I don't know based on the provided document."
"""

response = llm.invoke(prompt)

print(response.content)
```

This is the fundamental RAG pattern:

> **Retrieve relevant information first, then give that information to the
> LLM so it can generate an answer grounded in the retrieved context.**

---

# RAG Function

The retrieval and LLM steps were then combined into a reusable function:

```python
def ask_rag(question: str):
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    prompt = f"""
You are answering questions about a WHO document.

Use ONLY the context below.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context,
say that you don't know.
"""

    response = llm.invoke(prompt)

    return response.content
```

This allows us to query the RAG system with:

```python
answer = ask_rag(
    "What are the key infection prevention measures?"
)

print(answer)
```

---

# RAG as an Agent Tool

The final step we reached was connecting the RAG retrieval process to a
LangChain agent.

Instead of manually calling the RAG function, we expose retrieval as a
tool:

```python
from langchain_core.tools import tool

@tool
def search_who_document(query: str) -> str:
    """
    Search the WHO infection prevention and control
    document for information relevant to the user's question.
    """

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = results["documents"][0]

    return "\n\n".join(documents)
```

The agent can then decide when this tool is necessary:

```text
User
 ↓
Agent / LLM
 ↓
Decides whether information is needed
 ↓
RAG Tool
 ↓
Embedding Model
 ↓
ChromaDB
 ↓
Relevant Documents
 ↓
LLM
 ↓
Final Answer
```

This is different from manually calling the retriever because the agent
can choose the appropriate tool based on the user's question.

---

# Key RAG Concepts Learned

At this point, the complete RAG pipeline is understood as:

```text
                INGESTION
                    │
                    ▼
                  PDF
                    │
                    ▼
             Document Loader
                    │
                    ▼
                 Chunks
                    │
                    ▼
               Embeddings
                    │
                    ▼
                ChromaDB
                    │
                    │
                    ▼
                RETRIEVAL
                    │
              User Question
                    │
                    ▼
             Query Embedding
                    │
                    ▼
                ChromaDB
                    │
                    ▼
          Relevant Documents
                    │
                    ▼
                   LLM
                    │
                    ▼
                 Answer
```

The important separation is:

```text
Embedding model
    → converts text into vectors

Vector database
    → stores and searches vectors

Retriever
    → gets relevant documents

LLM
    → understands the question and generates the answer

Agent
    → decides when and which tool to use
```

---

# Current Progress

```text
[✓] SQLite data
[✓] Pydantic schemas
[✓] JSON document creation
[✓] JSONLoader
[✓] LangChain Documents
[✓] Custom metadata with metadata_func

[✓] Document splitting / chunking
[✓] Embeddings
[✓] Vector store / ChromaDB
[✓] Semantic retrieval
[✓] Metadata filtering
[✓] Retrieval testing
[✓] PDF loading
[✓] PDF chunking
[✓] PDF embeddings
[✓] PDF vector storage
[✓] PDF retrieval testing
[✓] LLM + retrieved context
[✓] Complete basic RAG pipeline
[✓] RAG as a LangChain tool
[✓] RAG tool connected to an agent

[ ] Improve retrieval quality
[ ] LangChain Retriever abstractions
[ ] Prompt templates for RAG
[ ] Source/page citations in answers
[ ] Hybrid search
[ ] Metadata-aware retrieval
[ ] Reranking
[ ] Production RAG architecture
```

# Current Project Understanding

The project has now demonstrated two different types of data:

### Structured data

```text
SQLite
  ↓
Pydantic
  ↓
JSON
  ↓
JSONLoader
  ↓
LangChain Documents
  ↓
Meaningful chunks
  ↓
Embeddings
  ↓
ChromaDB
```

### Unstructured data

```text
PDF
  ↓
PyPDFLoader
  ↓
LangChain Documents
  ↓
Text splitting
  ↓
Embeddings
  ↓
ChromaDB
```

Both ultimately use the same retrieval concept:

```text
Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
LLM
```

The major difference is **how the original data is converted into useful
chunks**.

For structured data, domain-specific boundaries such as an individual
order can be more meaningful.

For unstructured documents such as PDFs, text-based chunking is generally
more appropriate.

---

# Next Steps

The Stages cleared:

```text
[✓] SQLite data
[✓] Pydantic schemas
[✓] JSON document creation
[✓] JSONLoader
[✓] LangChain Documents
[✓] Custom metadata with metadata_func
[✓] Document splitting / chunking
[✓] Embeddings
[✓] Vector store
[✓] Retriever / vector search
[✓] Retrieval testing
[✓] LLM + retrieved context
[✓] Complete basic RAG pipeline
[✓] RAG as an agent tool

```

> **Current milestone: We can now take a document, convert it into
> embeddings, store it in a vector database, retrieve relevant information,
> give that information to an LLM, and expose the retrieval process as an
> agent tool.**
