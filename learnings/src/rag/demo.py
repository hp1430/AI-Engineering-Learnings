import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv(override=True)

os.environ.setdefault("LLM_MODEL", "openai/gpt-oss-20b")

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to your Groq API key.")

FACTS = [
    "Zorvia Coffee annual subscriptions can be refunded within 14 days of purchase.",
    "Monthly subscriptions can be cancelled at any time, but the current month is not refunded.",
    "Orders of 5 bags or more ship free.",
    "Zorvia roasts every Tuesday and Friday in Lisbon.",
    "Customer support is open 9:00 to 17:00 UTC, Monday to Friday.",
    "The Ember blend is a dark roast with notes of cocoa and orange peel.",
    "The Drift blend is a light roast from Ethiopia with notes of jasmine and lemon.",
    "Zorvia gift cards never expire.",
    "Beans are freshest for 30 days after the roast date if kept in a sealed bag.",
    "Wholesale accounts require a minimum order of 20 kg per month.",
    "In the referral program, both the referrer and the friend get one free bag.",
    "Orders placed before 12:00 UTC ship the same day.",
]


chat_model = init_chat_model(
    os.environ.get("LLM_MODEL"),
    model_provider="openai",
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url=os.environ.get("GROQ_API_URL"),
)

def to_documents(facts: list[str]) -> list[Document]:
    """Convert a list of facts to a list of Documents"""
    return [Document(page_content=fact, metadata={"id": i}) for i, fact in enumerate(facts)]

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Get the embeddings model"""
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", # 384D embeddings
        encode_kwargs={"normalize_embeddings": True}    # l2 normalization of embeddings
    )
    return embeddings_model

def build_store(facts: list[Document]) -> InMemoryVectorStore:
    """Build a vector store from a list of facts"""
    store = InMemoryVectorStore(embedding=get_embeddings_model())   # this creates an empty vector store using sentence-transformers/all-MiniLM-L6-v2 embeddings
    store.add_documents(facts)  # this adds the documents to the vector store i.e. document -> embedding -> stored in the vector store
    return store

v_store = build_store(to_documents(FACTS))  # build the vector store from the facts

QUESTION = "How long do I have to ask for a refund on my annual plan ?"

# top k closest chunks with their similarity scores

for chunk, score in v_store.similarity_search_with_score(QUESTION, k=3):
    print(f"Score: {score:.2f}")
    print(f"Chunk: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("--------------------------------------------------")

def answer(query, store, k=3):
    relevant_facts = store.similarity_search(query, k=k)
    context = "\n".join([fact.page_content for fact in relevant_facts])
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = chat_model.invoke(prompt)
    return response

print("Direct answer with rag: ", answer(QUESTION, v_store))