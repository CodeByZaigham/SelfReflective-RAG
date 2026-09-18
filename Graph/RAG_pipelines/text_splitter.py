from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

pdf_txt_splitter=RecursiveCharacterTextSplitter(
     chunk_size=1200,
     chunk_overlap=150
)
csv_splitter=RecursiveCharacterTextSplitter(
     chunk_size=300,
     chunk_overlap=50
)
ppt_splitter=RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=100
)

@traceable(name="creating pdf chunks")
def chunk_pdf_txt(doc):
     chunks=pdf_txt_splitter.split_documents(doc)
     return chunks

@traceable(name="creating csv chunks")
def chunk_csv(doc):
     chunks=csv_splitter.split_documents(doc)
     return chunks

@traceable(name="creating ppt chunks")
def chunk_ppt(doc):
     chunks=ppt_splitter.split_documents(doc)
     return chunks