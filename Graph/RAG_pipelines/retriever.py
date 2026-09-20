from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from langsmith import traceable
from LLM import get_llm

# @traceable(name="retrieving chunks from vector store")
def retrieve_embeddings(data:dict):
        query=data["query"]
        db=data["db"]
        retriever=db.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4}
        )

        query_variations=MultiQueryRetriever.from_llm(
            llm=get_llm(),
            retriever=retriever
        )

        result=query_variations.invoke(query)

        return result