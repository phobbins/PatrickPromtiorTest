import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'


URLs=[
    'https://www.promtior.ai/',
    'https://www.promtior.ai/service',
    'https://www.promtior.ai/use-cases',
    'https://www.promtior.ai/contacto'

]

loader = WebBaseLoader(URLs)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

chunks = text_splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings()

vector = FAISS.from_documents(chunks, embeddings)

retriever = vector.as_retriever()