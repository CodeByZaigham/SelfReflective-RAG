from edges import workflow
from pathlib import Path

initialstate={
     "doc_path":str(Path(__file__).parent.parent / "Pdfs/deep learning book.pdf"),
     "query":"explain me K-mean clustering Algorithm using example given in the book.",
     "revise_attempts":0,
     "retrieve_attempts":0
}

final_response=None

#values streams state snapshots, not individual tokens.
for message in workflow.stream(initialstate,stream_mode="values"):
     if "response" in message:
          final_response = message["response"]

print(final_response)