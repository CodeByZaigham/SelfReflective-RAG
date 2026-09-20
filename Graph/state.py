from typing import TypedDict,List,Dict,Annotated,Literal
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

     retrieval_needed:Literal["needed","not needed"]
     relevent_docs:List[Document]
     is_answer_supported:Literal["supported","not supported"]
     is_answer_usable: Literal["usable","not usable"]

     revise_attempts:int
     retrieve_attempts:int
