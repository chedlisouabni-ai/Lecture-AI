import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

class LectureVectorStore:
    """
    Store and retrieve lecture content using semantic search
    """
    
    def __init__(self, persist_dir="./lecture_db"):
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="lectures",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Vector store initialized: {self.collection.count()} items stored")
    
    def add_lecture_content(self, chunks, lecture_id):
        """
        Add processed lecture chunks to vector store
        
        chunks: List of dicts with 'content', 'source', 'modality', etc.
        lecture_id: Identifier for this lecture (e.g., "Lecture_1", "Module_2_Video")
        """
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['content'])
            
            documents.append(chunk['content'])
            embeddings.append(embedding.tolist())
            metadatas.append({
                'lecture_id': lecture_id,
                'source': chunk['source'],
                'modality': chunk['modality'],
                'chunk_id': chunk.get('chunk_id', i),
                'timestamp': chunk.get('timestamp', 0)
            })
            ids.append(f"{lecture_id}_chunk_{i}")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(chunks)} chunks from {lecture_id}")
    
    def search(self, query, n_results=5, lecture_filter=None):
        """
        Semantic search across all lectures
        
        query: Question or topic to search for
        n_results: Number of results to return
        lecture_filter: Optional - only search specific lecture
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Build filter if specified
        where_filter = None
        if lecture_filter:
            where_filter = {"lecture_id": lecture_filter}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def get_lecture_content(self, lecture_id):
        """Retrieve all content from a specific lecture"""
        results = self.collection.get(
            where={"lecture_id": lecture_id}
        )
        return results