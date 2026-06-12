import os
import chromadb
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
import google.generativeai as genai

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, context: str, query: str) -> str:
        pass

class LocalLLM(LLMProvider):
    def generate(self, context: str, query: str) -> str:
        return f"[Local Model Active] based on the provided CV chunks (Length: {len(context)} chars), the answer to '{query}' is generated here."

class RemoteAPI_LLM(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash') 

    def generate(self, context: str, query: str) -> str:
        prompt = f"""
        You are an expert HR assistant tasked with answering queries based ONLY on the provided candidate CV snippets.
        
        Rules:
        1. Base your answer strictly on the provided context.
        2. Pay close attention to the "Candidate" and "Section" labels these are meteadata attached to each snippet to distinguish between different people and their skills.
        3. If the answer cannot be found in the context, reply exactly with: "I cannot answer this based on the provided CV data." Do not guess or hallucinate.
        4. Keep your answer concise and professional.
        
        Context Data:
        {context}
        
        User Query: 
        {query}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Remote LLM: {str(e)}"

class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str) -> LLMProvider:
        if provider_name.lower() == "local":
            return LocalLLM()
        elif provider_name.lower() == "remote":
            return RemoteAPI_LLM()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")


print("loading embedding model and connecting to ChromaDB")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')

chroma_client = chromadb.PersistentClient(path="./cv_chroma_db")
collection = chroma_client.get_collection(name="candidates")

ACTIVE_LLM = os.getenv("LLM_PROVIDER", "local")
llm_engine = LLMFactory.get_llm(ACTIVE_LLM)

def process_rag_query(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(f"query: {query}").tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results['documents'] or not results['documents'][0]:
        return "No relevant CV data found for this query.", []
        
    retrieved_documents = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0] # for meta data extraction

 
    formatted_chunks = []
    
    for doc, meta in zip(retrieved_documents, retrieved_metadatas):
        candidate = meta.get("candidate_name", "Unknown Candidate")
        section = meta.get("section", "Unknown Section")
        
        chunk_string = f"Candidate: {candidate}\nSection: {section}\nContent:\n{doc}"
        formatted_chunks.append(chunk_string)

  
    context_block = "\n\n-------------------------\n\n".join(formatted_chunks)
    print(f"your context block is as follows \n {context_block}")

    
    generated_answer = llm_engine.generate(context=context_block, query=query)
    return generated_answer, retrieved_documents