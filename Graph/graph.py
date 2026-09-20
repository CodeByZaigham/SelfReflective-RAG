from edges import workflow
from pathlib import Path

initialstate={
     "doc_path":str(Path(__file__).parent.parent / "Pdfs/deep learning book.pdf"),
     "query":"explain me K-mean clustering Algorithm using example given in the book.",
     "revise_attempts":0,
     "retrieve_attempts":0
}

response=workflow.invoke(initialstate)
print(response["response"])