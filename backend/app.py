from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils.embedding_functions.instructor_embedding_function import InstructorEmbeddingFunction
import re
from pydantic import BaseModel

class TextPayload(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Allow your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

ef = InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
client = chromadb.PersistentClient()
collection1 = client.get_or_create_collection(name="chapter", metadata={"hnsw:space": "cosine"}, embedding_function= ef)
collection = client.get_or_create_collection(name="hscode", metadata={"hnsw:space": "cosine"}, embedding_function=ef)

def preprocess_text(text):
    # Your preprocessing steps here
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace("<il>", "").replace("</il>", "")
    text = re.sub(r'[;/:<>,()]', ' ', text)
    text = re.sub(r"\S*https?:\S*", '', text)
    text = re.sub("[''“”‘’…]", '', text)
    # Additional preprocessing steps if needed
    return text
@app.post("/query/")
async def query_hs_code(payload: TextPayload):
    text_received = payload.text
    
    if not text_received:
        raise HTTPException(status_code=400, detail="No query text provided")

    results = collection1.query(query_texts=[text_received], n_results=3)
    similar_word_check = collection1.query(query_texts=[text_received], n_results=1)
   
    output = []
    for ids, distances, metadatas in zip(results['ids'], results['distances'], results['metadatas']):
      for id, distance, metadata in zip(ids, distances, metadatas):
        try:
          chapter_code = metadata['HS CODE']
          if len(chapter_code) == 4:
            hscode_result= collection.query(
                  query_texts=[text_received],
                  n_results=3,
                  where={
                  "checker": {
                      "$eq": chapter_code
                        }
                    }
              )
            for hs_ids, hs_distances, hs_metadatas in zip(hscode_result['ids'], hscode_result['distances'], hscode_result['metadatas']):
              for hs_id, hs_distance, hs_metadata in zip(hs_ids, hs_distances, hs_metadatas):
                chapter_code = hs_metadata['HS CODE']
                output.append({"HS Code": chapter_code, "Score": round((1 - hs_distance) *100, 2),"chap_desc":metadata['DESCRIPTION'], "desc": hs_metadata['DESCRIPTION']})
          else:
            output.append({"HS Code": chapter_code, "Score": round((1 - distance) *100, 2), "chap_desc":metadata['DESCRIPTION'], "desc": ""})
        except:
          continue
    
    for metadatas in similar_word_check['metadatas']:
      for metadata in metadatas:
        try:
          chapter_code = metadata['HS CODE'][:4]
          hscode_result= collection.query(
                query_texts=[text_received],
                n_results=1,
                where={
                "checker": {
                    "$eq": chapter_code
                      }
                  }
            )
          for hs_ids, hs_distances, hs_metadatas in zip(hscode_result['ids'], hscode_result['distances'], hscode_result['metadatas']):
            for hs_id, hs_distance, hs_metadata in zip(hs_ids, hs_distances, hs_metadatas):
              chapter_code = hs_metadata['HS CODE']
              output.append({"HS Code": chapter_code, "Score": round((1 - hs_distance) *100, 2), "chap_desc":metadata['DESCRIPTION'], "desc": hs_metadata['DESCRIPTION']})
        except:
          continue
    seen = set()
    unique_output = []
    for entry in output:
      if entry['HS Code'] not in seen:
          seen.add(entry['HS Code'])
          unique_output.append(entry)
    formatted_results = sorted(unique_output, key=lambda x: x['Score'], reverse=True)
    return formatted_results

