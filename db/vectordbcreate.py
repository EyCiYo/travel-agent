import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("openai_key")

current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "Shengen_faq.pdf")
txtpath = os.path.join(current_dir, "visa_details.txt")
persistent_dir = os.path.join(current_dir, "vector_store")

if not os.path.exists(persistent_dir):
    print("Creating new persistant directory..")
    if not os.path.exists(filepath):
        raise FileNotFoundError("cant find file")

    pdfLoader = PyPDFLoader(file_path=filepath)
    pdfDocs = pdfLoader.load()

    textLoader = TextLoader(file_path=txtpath)
    txtDocs = textLoader.load()

    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10)
    allDocs = pdfDocs + txtDocs
    docs = textSplitter.split_documents(documents=allDocs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=apikey)
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir)
else:
    print("already exist")
