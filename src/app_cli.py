import argparse
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq

from settings import GROQ_API_KEY, GROQ_MODEL, TOP_K
from prompts import SYSTEM_PROMPT, ANSWER_FORMAT_HINT
from utils import load_faiss_index

def build_chain():
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in .env")

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.2)
    vectordb = load_faiss_index()
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\n" + ANSWER_FORMAT_HINT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])

    def format_docs(docs):
        # include short source list at top
        sources = list({d.metadata.get("source", "Wikipedia") for d in docs})
        header = "Sources: " + ", ".join(sources) + "\n\n"
        bodies = "\n\n---\n\n".join(d.page_content[:1200] for d in docs)
        return header + bodies

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

def ask(question: str):
    chain = build_chain()
    resp = chain.invoke(question)
    return resp.content if hasattr(resp, "content") else str(resp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Wikipedia RAG (LangChain + Groq)")
    parser.add_argument("question", type=str, help="Your question about deep learning")
    args = parser.parse_args()

    print(ask(args.question))
