import os
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from utils import is_answer_confident

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is required. Set it in your .env")

if not PINECONE_ENVIRONMENT:
    raise RuntimeError("PINECONE_ENVIRONMENT is required. Set it in your .env")


qa_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "Sorry, I don't know. Please try some other query."
Provide as much detail as possible from the context.

Context:
{context}

Question: {question}
Answer:
"""
)


# def _init_pinecone():
#     pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#     return pc


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return vector_store


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
        combine_docs_chain_kwargs={"prompt": qa_prompt},
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
                "retrieved_chunks": [],
            }

        result = chain({"question": question})
        answer = result["answer"]
        confident, score = is_answer_confident(answer)

        return {
            "answer": answer if confident else "I'm not sure.",
            "confidence_score": score,
            "retrieved_chunks": (
                [doc.page_content[:500]
                    for doc in filtered_docs] if confident else []
            ),
        }

    return guarded_chain
