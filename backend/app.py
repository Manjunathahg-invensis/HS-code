from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils.embedding_functions.instructor_embedding_function import InstructorEmbeddingFunction
import re
from pydantic import BaseModel

class TextPayload(BaseModel):
    text: str

app = FastAPI()

# CORS Middleware: allow all origins for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize ChromaDB client and collections
ef = InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
client = chromadb.PersistentClient()
collection1 = client.get_or_create_collection(
    name="chapter",
    metadata={"hnsw:space": "cosine"},
    embedding_function=ef
)
collection = client.get_or_create_collection(
    name="hscode",
    metadata={"hnsw:space": "cosine"},
    embedding_function=ef
)

# Preprocessing function for input text
def preprocess_text(text):
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace("<il>", "").replace("</il>", "")
    text = re.sub(r'[;/:<>,()]', ' ', text)
    text = re.sub(r"\S*https?:\S*", '', text)
    text = re.sub("[''“”‘’…]", '', text)
    return text

# POST route for querying HS code
@app.post("/query/")
async def query_hs_code(payload: TextPayload):
    text_received = payload.text
    
    if not text_received:
        raise HTTPException(status_code=400, detail="No query text provided")

    # Try querying the first collection (chapter)
    try:
        results = collection1.query(query_texts=[text_received], n_results=3)
        similar_word_check = collection1.query(query_texts=[text_received], n_results=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying chapter collection: {str(e)}")

    output = []
    
    # Process results from the chapter collection
    for ids, distances, metadatas in zip(results['ids'], results['distances'], results['metadatas']):
        for id, distance, metadata in zip(ids, distances, metadatas):
            try:
                chapter_code = metadata.get('HS CODE', '')
                if len(chapter_code) == 4:
                    hscode_result = collection.query(
                        query_texts=[text_received],
                        n_results=3,
                        where={"checker": {"$eq": chapter_code}}
                    )
                    for hs_ids, hs_distances, hs_metadatas in zip(hscode_result['ids'], hscode_result['distances'], hscode_result['metadatas']):
                        for hs_id, hs_distance, hs_metadata in zip(hs_ids, hs_distances, hs_metadatas):
                            chapter_code = hs_metadata.get('HS CODE', '')
                            output.append({
                                "HS Code": chapter_code,
                                "Score": round((1 - hs_distance) * 100, 2),
                                "chap_desc": metadata.get('DESCRIPTION', ''),
                                "desc": hs_metadata.get('DESCRIPTION', '')
                            })
                else:
                    output.append({
                        "HS Code": chapter_code,
                        "Score": round((1 - distance) * 100, 2),
                        "chap_desc": metadata.get('DESCRIPTION', ''),
                        "desc": ""
                    })
            except Exception as e:
                # Log or handle specific errors
                continue
    
    # Process similar word check results
    try:
        for metadatas in similar_word_check['metadatas']:
            for metadata in metadatas:
                chapter_code = metadata.get('HS CODE', '')[:4]
                hscode_result = collection.query(
                    query_texts=[text_received],
                    n_results=1,
                    where={"checker": {"$eq": chapter_code}}
                )
                for hs_ids, hs_distances, hs_metadatas in zip(hscode_result['ids'], hscode_result['distances'], hscode_result['metadatas']):
                    for hs_id, hs_distance, hs_metadata in zip(hs_ids, hs_distances, hs_metadatas):
                        chapter_code = hs_metadata.get('HS CODE', '')
                        output.append({
                            "HS Code": chapter_code,
                            "Score": round((1 - hs_distance) * 100, 2),
                            "chap_desc": metadata.get('DESCRIPTION', ''),
                            "desc": hs_metadata.get('DESCRIPTION', '')
                        })
    except Exception as e:
        # Handle any issues during the similar word check phase
        pass

    # Remove duplicates and sort by score
    seen = set()
    unique_output = []
    for entry in output:
        if entry['HS Code'] not in seen:
            seen.add(entry['HS Code'])
            unique_output.append(entry)

    formatted_results = sorted(unique_output, key=lambda x: x['Score'], reverse=True)
    return formatted_results
