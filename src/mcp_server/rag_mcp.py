import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

class DocumentRAG:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.db_path = self.base_path / "data" / "chroma_db"
        self.policy_path = self.base_path / "data" / "policies" / "it_corporate_policy.md"
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ GEMINI_API_KEY not found in .env file")
            
        self.client = genai.Client(api_key=self.api_key)
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_or_create_collection(name="corporate_policies")
        
        if self.collection.count() == 0:
            print("[RAG] Database empty. Indexing corporate policies...")
            self._ingest_initial_documents()

    def _get_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return response.embeddings[0].values

    def _ingest_initial_documents(self):
        if not self.policy_path.exists():
            return
        with open(self.policy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.ingest_new_document(content, "it_corporate_policy.md")

    def ingest_new_document(self, content: str, source_name: str) -> int:
        """Splits and ingests a new document WITH source metadata."""
        chunks = content.split("\n## ")
        chunks_added = 0
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            if i > 0:
                chunk = "## " + chunk
                
            vector = self._get_embedding(chunk)
            
            doc_id = f"{source_name}_chunk_{i}_{hash(chunk)}"
            self.collection.add(
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": source_name}],
                ids=[doc_id]
            )
            chunks_added += 1
            
        return chunks_added

    def search_knowledge_base(self, query: str, top_k: int = 1) -> str:
        query_vector = self._get_embedding(query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documentation found."
            
        # Format the output to explicitly include the [Source: filename]
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            # Added robustness: fallback if legacy document lacks metadata.
            if meta is None:
                source_file = "Unknown Policy"
            else:
                source_file = meta.get('source', 'Unknown Policy')
                
            context_parts.append(f"--- [SOURCE DOCUMENT: {source_file}] ---\n{doc}")
            
        return "\n\n".join(context_parts)