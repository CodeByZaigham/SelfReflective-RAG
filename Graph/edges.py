from langgraph.graph import StateGraph,START,END
from nodes import (
     load_document,
     create_database,
     retriever,
     generate_from_context,
     generate_direct,
     rewrite_query,
     is_answer_supported,
     is_answer_usable,
     revise_answer,
     docs_relevence,
     decide_retrieval
)
from state import state

graph=StateGraph(state)

graph.add_node("load_document",load_document)
graph.add_node("vector_database",create_database)
graph.add_node("retriever",retriever)
graph.add_node("response_generator_from_context",generate_from_context)
graph.add_node("generate_direct",generate_direct)
graph.add_node("rewrite_query",rewrite_query)
graph.add_node("is_answer_supported",is_answer_supported)
graph.add_node("is_answer_usable",is_answer_usable)
graph.add_node("revise_answer",revise_answer)
graph.add_node("docs_relevence",docs_relevence)
graph.add_node("decide_retrieval",decide_retrieval)

def retrieval_needed(state:state):
     if state["retrieval_needed"] == "needed":
          return "load_document"
     else: return "generate_direct"

def is_relevent(state:state):
     if state["relevent_docs"]:
          return "response_generator_from_context"
     else: END

def is_factual(state:state):
     if state["is_answer_supported"] == "supported":
          return "is_answer_usable"
     else: "revise_answer"

def is_answer_complete(state:state):
     if state["is_answer_usable"] == "usable":
          return END
     elif state["is_answer_usable"] == "not usable":
          state["retrieve_attempts"] += 1 
          return ""


graph.add_edge(START,"decide_retrieval")
graph.add_conditional_edges("decide_retrieval",retrieval_needed)
graph.add_edge("load_document","vector_database")
graph.add_edge("vector_database","retriever")


graph.add_edge("retriever",is_relevent)

graph.add_edge("generate_direct",END)
graph.add_edge("response_generator_from_context","is_answer_supported")
graph.add_edge("is_answer_supported",is_factual)

graph.add_node("is_answer_usable",is_answer_complete)

workflow=graph.compile()