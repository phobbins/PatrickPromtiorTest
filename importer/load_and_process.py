import os

from langchain_community.document_loaders import WebBaseLoader
from lanchain_community.embeddings import OpenAIEmbeddings

from config import EMBEDDING_MODEL

URLs=[
    'https://www.promtior.ai/',
    'https://www.promtior.ai/service',
    'https://www.promtior.ai/use-cases',
    'https://www.promtior.ai/contacto'

]

loader = WebBaseLoader(URLs)
docs = loader.load()
print(docs)

OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

