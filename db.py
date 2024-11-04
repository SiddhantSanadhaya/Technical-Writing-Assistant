# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# import os


# # Path to the PDF and database
# pdf_path = r"C:\Users\DESKTOP\Downloads\TIBCO Flogo® Enterprise User Guide.pdf"
# db_path = "path_to_saved_db_lamma_3.2"  # Specify path to save the database

# # Load and process PDF
# loader = PyPDFLoader(pdf_path)
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# documents = text_splitter.split_documents(docs)
# print(f"Number of document chunks: {len(documents)}")

# # Initialize the embedding model
# embedding_model = OllamaEmbeddings(model="llama3.2")

# # Initialize or load the Chroma vector store
# if os.path.exists(db_path):
#     db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
# else:
#     db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

# # Process documents in batches
# batch_size = 20
# for i in range(0, len(documents), batch_size):
#     batch = documents[i:i+batch_size]
#     texts = [doc.page_content for doc in batch]
#     metadatas = [doc.metadata for doc in batch]
    
#     db.add_texts(texts=texts, metadatas=metadatas)
#     print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

# # Persist the database
# db.persist()

# print(f"Database successfully saved at {db_path}")


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
key = os.environ['key']

from langchain_community.llms import OpenAI

# Initialize GPT-4 or GPT-3.5 Turbo
llm = OpenAI(model="gpt-4",openai_api_key = key,)

# Path to the PDF and database
pdf_path = r"C:\Users\DESKTOP\Downloads\TIBCO Flogo® Enterprise User Guide.pdf"
db_path = "path_to_saved_db_gpt_4_chunk_1000"  # Specify path to save the database

# Load and process PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
print(f"Number of document chunks: {len(documents)}")

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=key,)

# Initialize or load the Chroma vector store
# if os.path.exists(db_path):
#     db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
# else:
db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

# Process documents in batches
batch_size = 20
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    
    db.add_texts(texts=texts, metadatas=metadatas)
    print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

# Persist the database
db.persist()

print(f"Database successfully saved at {db_path}")