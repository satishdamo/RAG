from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from utils import is_answer_confident
load_dotenv()


VECTOR_DB_ROOT = os.environ.get("VECTOR_DB_ROOT", "chroma_db")

qa_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "Sorry, I don't know. Please try some other query."
Provide as much detail as possible from the context.

Context:
{context}

Question: {question}
Answer:
""")


def load_vectorstore():
    embedding = OpenAIEmbeddings()
    return Chroma(persist_directory=VECTOR_DB_ROOT, embedding_function=embedding)


def build_qa_chain_with_memory():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    def guarded_chain(inputs):
        question = inputs["question"]
        retrieved_docs = retriever.get_relevant_documents(question)
        filtered_docs = [
            doc for doc in retrieved_docs if doc.page_content.strip()]

        if not filtered_docs:
            return {
                "answer": "I'm not sure.",
                "confidence_score": 0.0,
                "retrieved_chunks": []
            }

        result = chain({"question": question})
        answer = result["answer"]
        confident, score = is_answer_confident(answer)

        return {
            "answer": answer if confident else "I'm not sure.",
            "confidence_score": score,
            "retrieved_chunks": [doc.page_content[:500] for doc in filtered_docs] if confident else []
        }

    return guarded_chain
