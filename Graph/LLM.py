from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

def get_llm():
     return ChatGroq(model="openai/gpt-oss-120b")