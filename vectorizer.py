import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from chunker import semantic_chunk_cv

def build_vector_db(raw_cv_texts):
    print("loading embedding model in memory")
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    print("init chromaDB")
    chroma_client = chromadb.PersistentClient(path="./cv_chroma_db")
    
    collection = chroma_client.get_or_create_collection(name="candidates")
    
    print("processing and inserting cvs")
    for cv_text in raw_cv_texts:
       
        chunks = semantic_chunk_cv(cv_text)
        

        for chunk in chunks:
            text = chunk['text'] # the actual content of dictionary of this chunk
            metadata = chunk['metadata'] # chunk title and person_name of related to this chunk
            print(f"---- meta of chunk {metadata} ---- \n")
            print(f"the data of chunk --- \n {text}")
            
			# generate unique id for this vector
            chunk_id = str(uuid.uuid4()) 
            embedding = model.encode(f"passage: {text}").tolist() 
            
			# act as inserting a row in chromaDB
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
    print(f"success store chunks in the vector database")
