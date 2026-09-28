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

Used to define a predictable structure for our data before creating JSON
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

``` python
jq_schema=".[]"
```

means each customer becomes one document.

### `page_content`

Contains the actual document content.

### `metadata`

Contains small attributes describing the document.

### `metadata_func`

Allows us to customize the metadata generated by `JSONLoader`.

------------------------------------------------------------------------

# Next Steps

These will be added as the RAG project progresses.

``` text
[✓] SQLite data
[✓] Pydantic schemas
[✓] JSON document creation
[✓] JSONLoader
[✓] LangChain Documents
[✓] Custom metadata with metadata_func
[ ] Document splitting / chunking
[ ] Embeddings
[ ] Vector store
[ ] Retriever
[ ] Retrieval testing
[ ] LLM + retrieved context
[ ] Complete RAG pipeline
```

The next topic to explore is:

> **Document splitting / chunking**

Before moving to embeddings, we need to understand why a document may
need to be split into smaller pieces and how metadata should be
preserved when that happens.
