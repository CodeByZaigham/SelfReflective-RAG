from edges import workflow
from pathlib import Path

initialstate={
     "doc_path":str(Path(__file__).parent.parent / "Pdfs/deep learning book.pdf"),
     "query":"what is deep learning"
}

response=workflow.invoke(initialstate)
print(response["response"])