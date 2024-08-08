import json
import ollama
import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
from datetime import datetime

EMBEDDING_MODEL = 'llama3.1'


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return ollama.embed(model=EMBEDDING_MODEL, input=input)['embeddings']


class ChatMemory():
    def __init__(self, chat_id: str, collection_name='llm_mem', db_path='./llm_mem', distance_threshold=1) -> None:
        self.collection_name = collection_name
        self.db_path = db_path
        self.chat_id = chat_id
        self.distance_threshold = distance_threshold

        self.included_memory_ids = []
        """list of ids where that memory is already included into the chat. So including it multiple times could waste the token limit + confuse the model"""

        # number of chat objects saved from the current chat
        # used to index the chat object when saving so it's easier when retrieving later
        self.n_saved_from_this_chat = 0

        self._chroma_client = chromadb.PersistentClient(path=self.db_path)

        try:
            self._load_db()
        except ValueError:
            self._create_db()

    def _load_db(self):
        self.db = self._chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=OllamaEmbeddingFunction()
        )

    def _create_db(self):
        self.db = self._chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=OllamaEmbeddingFunction()
        )

    def get_relevant_memories(self, query: str, n_results=5):
        related_memories_from_db = self.db.query(
            query_texts=query,
            n_results=n_results,
            where={
                "chat_id": {
                    "$ne": self.chat_id
                }
            }
        )

        related_memories = []

        for i, id in enumerate(related_memories_from_db['ids'][0]):

            # Not including the memory if it already has been included into the chat. including multiple times could waste the token limit.
            if id in self.included_memory_ids:
                continue

            # Memories that are returned after querying are ordered on the closest to the prompt. But sometimes they are no use to the prompt so they get removed with a distance threshold
            # Loop needs to break when it finds a memory that is distant than the threshold because memories after that are def not within the threshold. (CUZ THEY IN ORDER)
            if self.distance_threshold < related_memories_from_db['distances'][0][i]:
                break

            memory = {
                "id": id,
                "content": related_memories_from_db['documents'][0][i],
                "metadata": related_memories_from_db['metadatas'][0][i],
            }
            self.included_memory_ids.append(id)
            related_memories.append(memory)

        return related_memories

    def save_chat_obj(self, chat_obj: dict):
        self.n_saved_from_this_chat = self.n_saved_from_this_chat + 1
        memory_metadata = {
            "created_at": str(datetime.now()),
            "chat_id": self.chat_id,
            "n_message": self.n_saved_from_this_chat,
            "role": chat_obj['role'],
        }
        id = f"{self.chat_id}:{self.n_saved_from_this_chat}"
        self.n_saved_from_this_chat

        self.db.add(
            ids=id,
            documents=chat_obj['content'],
            metadatas=memory_metadata
        )
