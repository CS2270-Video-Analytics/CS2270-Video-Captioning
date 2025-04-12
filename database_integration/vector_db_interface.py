import os
from config import Config
import faiss
import torch
import torch.nn.functional as F
import pdb

class VectorDBInterface():

    def __init__(self, vector_dim:torch.Tensor, db_name:str = None, save_path:str = None):

        #store the DB name
        self.db_name = db_name if db_name is not None else Config.vec_db_name

        #path to saved vector DB
        self.db_path = os.path.join(save_path if save_path is not None else Config.vec_db_path, self.db_name)

        #create the vector index during the init
        self.create_index(vector_dim = vector_dim)
    

    def create_index(self, vector_dim:int):

        if os.path.exists(self.db_path):
            self.load_vectordb()
        else:
            self.vector_index = faiss.IndexFlatIP(vector_dim)
        
        self.save_vectordb()

    def seach_query(self, query: torch.Tensor, k:int = 1):
        self.load_vectordb()
        query = F.normalize(query, p=2, dim=1)   # Normalize the query vector

        distances, indices = self.vector_index.search(query, k)
        self.save_vectordb()
        self.vector_index = None

        return distances, indices
    
    def search_all(self, query: torch.Tensor):
        self.load_vectordb()
        query = F.normalize(query, p=2, dim=1)   # Normalize the query vector

        distances, indices = self.vector_index.search(query, k = self.vector_index.ntotal)
        self.save_vectordb()
        self.vector_index = None
        return distances, indices
    
    def get_num_vecs(self):
        self.load_vectordb()
        num_vecs = self.vector_index.ntotal
        self.save_vectordb()
        self.vector_index = None
        return num_vecs

    def insert_many_vectors(self, vectors: torch.Tensor):
        self.load_vectordb()
        # Insert multiple vectros at once: normalize first so inner-product = cosine similarity for CLIP embeddings
        vectors = F.normalize(vectors, p=2, dim=1)
        self.vector_index.add(vectors)
        self.save_vectordb()
        self.vector_index = None

    def load_vectordb(self):
        self.vector_index = faiss.read_index(self.db_path)
    
    def save_vectordb(self):

        faiss.write_index(self.vector_index, self.db_path)


if __name__ == '__main__':

    import clip

    dummy = VectorDBInteface()
