from langgraph.graph import StateGraph,START,END
from nodes import load_document , create_database , retriever , generate_response
from state import state

graph=StateGraph(state)

graph.add_node("load_document",load_document)
graph.add_node("vector_database",create_database)
graph.add_node("retriever",retriever)
graph.add_node("response_generator",generate_response)

graph.add_edge(START,"load_document")
graph.add_edge("load_document","vector_database")
graph.add_edge("vector_database","retriever")
graph.add_edge("retriever","response_generator")
graph.add_edge("response_generator",END)

workflow=graph.compile()