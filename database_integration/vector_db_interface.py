import sys
import os
sys.path.append('..')
from config import Config
import faiss
import torch

class VectorDBInteface():

    def __init__(self, db_name:str = None, table_name:str = None, caption_schema:str = None, save_path:str = None):

        #store the DB name
        self.db_name = db_name

        #path to saved vector DB
        self.db_path = save_path

        #create the vector index during the init
        self.create_index()
    

    def create_index(self, sample_vector:torch.Tensor):

        if os.path.exists(self.db_path):
            self.load_vectordb()
        else:
            self.vector_index = faiss.IndexFlatL2(sample_vector.shape[-1])

    def seach_query(self, query: torch.Tensor, k:int = 1):
        
        faiss.normalize_L2(query)  # Normalize the query vector

        distances, indices = self.vector_index.search(query, k)

        return distances, indices
    
    def search_all(self, query: torch.Tensor):
        
        faiss.normalize_L2(query)  # Normalize the query vector

        distances, indices = self.vector_index.search(query, k = self.vector_index.ntotal)

        return distances, indices

    def insert_many_vectors(self, vectors: torch.Tensor):
        
        # Insert multiple vectros at once: normalize first so L2 = cosine similarity for CLIP embeddings
        faiss.normalize_L2(vectors)


        self.vector_index.add(vectors)

    def load_vectordb(self):
        self.vector_index = faiss.read_index(self.db_path)
    
    def save_vectordb(self):

        faiss.write_index(self.vector_index, self.db_path)