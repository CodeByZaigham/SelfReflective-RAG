from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,UnstructuredPowerPointLoader
from .text_splitter import chunk_pdf_txt,chunk_csv,chunk_ppt
from langsmith import traceable
import os

@traceable(name="loading text file")
def load_text_file(path:str):
     doc=TextLoader(path)
     text=doc.load()
     return text

@traceable(name="loading pdf")
def load_pdf_file(path:str):
     docs= PyPDFLoader(path)
     Text=docs.load()
     return Text

@traceable(name="loading csv")
def load_csv_file(path: str):
     doc=CSVLoader(file_path=path, encoding="utf-8")
     text=doc.load()
     return text

@traceable(name="loading ppt")
def load_ppt_file(path: str):
     doc=UnstructuredPowerPointLoader(path)
     text=doc.load()
     return text

@traceable(name="loading document")
def load_file(path: str):
     ext = os.path.splitext(path)[1].lower()

     if ext == ".pdf":
          docs=load_pdf_file(path)
          return chunk_pdf_txt(docs)
     elif ext == ".csv":
          docs= load_csv_file(path)
          return chunk_csv(docs)
     elif ext in [".ppt", ".pptx"]:
          docs=load_ppt_file(path)
          return chunk_ppt(docs)
     elif ext == ".txt":
          docs=load_text_file(path)
          return chunk_pdf_txt(docs)
     else:
          raise ValueError("Unsupported file type")