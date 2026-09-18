from typing import TypedDict,List,Dict,Annotated
from langgraph.graph import add_messages
from langchain_core.documents import Document
from langchain_chroma import Chroma

class state(TypedDict):
     doc_path:str
     query:str
     chunks:List[Document]
     database:Chroma
     retrieved_docs:List[Document]
     response:str
