# gemini.py

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
result = model.invoke("Write me a ballad about LangChain")

print(result.text)