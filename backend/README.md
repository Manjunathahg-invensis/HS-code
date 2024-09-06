These is explanation for the code in app.py 

1. Imports and Setup

from fastapi import FastAPI, Request, HTTPException
from typing import List
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import re
from pydantic import BaseModel

FastAPI: A web framework for building APIs with Python.
Request, HTTPException: Used to handle incoming HTTP requests and raise exceptions.
List: A type hint for lists.
chromadb: A library for working with embeddings, specifically for creating and querying collections of embeddings.
embedding_functions: Contains utilities for embedding text.
re: Python's regular expression module for text processing.
BaseModel: A Pydantic model, which is used to validate and structure data.

2. Defining a Request Model

class TextPayload(BaseModel):
    text: str
TextPayload: A data model representing the structure of the incoming JSON payload. It expects a single field, text, which must be a string.

3. Initializing FastAPI Application

app = FastAPI()
app: An instance of FastAPI, which is the main application object used to define routes and handle requests.

4. Embedding Function and ChromaDB Client Initialization

ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
ef: Creates an embedding function using the "hkunlp/instructor-xl" model. This function will be used to generate embeddings from text input.

client = chromadb.PersistentClient()
client: Initializes a ChromaDB persistent client, which allows for managing and querying embeddings stored in the database.

5. Creating/Retrieving Collections

collection1 = client.get_or_create_collection(name="chapter", metadata={"hnsw:space": "cosine"}, embedding_function= ef)
collection = client.get_or_create_collection(name="hscode", metadata={"hnsw:space": "cosine"}, embedding_function=ef)
collection1: Creates or retrieves a collection named "chapter" in the database. It uses cosine similarity to compare embeddings.
collection: Similarly, creates or retrieves a collection named "hscode". Both collections use the embedding function defined earlier.

6. Preprocessing Function

def preprocess_text(text):
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace("<il>", "").replace("</il>", "")
    text = re.sub(r'[;/:<>,()]', ' ', text)
    text = re.sub(r"\S*https?:\S*", '', text)
    text = re.sub("[''“”‘’…]", '', text)
    return text
preprocess_text: This function processes the input text to make it suitable for embedding:
Converts the text to lowercase.
Removes specific HTML-like tags (<il> and </il>).
Replaces punctuation marks with spaces.
Removes URLs and special characters.

7. API Endpoint Definition

@app.post("/query/")
async def query_hs_code(payload: TextPayload):
query_hs_code: This is the main endpoint for querying HS codes. It accepts a POST request at the /query/ URL and takes a TextPayload object as input.

8. Extracting and Validating Input Text

    text_received = payload.text
    
    if not text_received:
        raise HTTPException(status_code=400, detail="No query text provided")
text_received: Extracts the text field from the incoming request.
Validation: Checks if the text is empty. If so, it raises an HTTP 400 error indicating that no text was provided.

9. Querying the chapter Collection
    results = collection1.query(query_texts=[text_received], n_results=3)
    similar_word_check = collection1.query(query_texts=[text_received], n_results=1)
results: Queries the chapter collection for the top 3 most similar chapters to the input text.
similar_word_check: Queries the chapter collection for the single most similar chapter to check for very closely related terms.

10. Processing Query Results

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
output: An empty list to store the final results.

Loop through results:

The code iterates over the query results, extracting the ids, distances, and metadatas.
chapter_code: Extracts the HS code from the metadata.
If length of chapter_code is 4: It queries the hscode collection to find more specific HS codes related to the chapter code.
Appending Results: The resulting HS codes and their descriptions are added to the output list, along with a similarity score calculated from the distance.
Error Handling: If any error occurs during the processing of a particular result, it is caught, and the loop continues with the next result.

11. Similar Word Check Processing

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
Similar Word Check:
This loop processes the result of similar_word_check, which focuses on the most similar chapter.
It tries to extract the first 4 characters of the chapter code and uses that to query the hscode collection.
The relevant HS codes, scores, and descriptions are added to the output list.

12. Removing Duplicate Results
    seen = set()
    unique_output = []
    for entry in output:
      if entry['HS Code'] not in seen:
          seen.add(entry['HS Code'])
          unique_output.append(entry)
seen: A set that keeps track of HS codes that have already been added to the final output.
unique_output: A list that stores only unique HS codes by checking against the seen set.

13. Sorting and Returning the Results
    formatted_results = sorted(unique_output, key=lambda x: x['Score'], reverse=True)
    return formatted_results
Sorting: The unique results are sorted by their similarity score in descending order.
Return: The sorted list of HS codes and their descriptions is returned as the response to the API request.


This code provides an API that takes a text input, processes it to find the most relevant HS codes using embeddings stored in a ChromaDB, and returns these codes along with similarity scores and descriptions. The results are filtered to remove duplicates and are sorted by relevance before being returned to the client.


These is explanation for the code in bulk.py

1. Importing Required Libraries

import concurrent.futures
import pandas as pd
import requests
import time
from tqdm import tqdm
concurrent.futures: A module that provides a high-level interface for asynchronously executing tasks using threads or processes.
pandas as pd: A library for data manipulation and analysis, often used for handling tabular data (like CSV files).
requests: A library for making HTTP requests in Python, used here to send requests to an API.
time: A module that provides time-related functions, such as sleep.
tqdm: A library used to display progress bars in loops, useful for tracking the progress of operations.

2. Delaying Execution
time.sleep(15)
time.sleep(15): Pauses the execution of the script for 15 seconds. This might be used to ensure that the API server (running on localhost:8000) is up and running before the script starts making requests.

3. Loading and Initializing Data
data = pd.read_csv("test.csv", encoding="latin1")
data['Q1'] = None
data['Q2'] = None
data['Q3'] = None
url = "http://localhost:8000/query/"
pd.read_csv("test.csv", encoding="latin1"): Loads the CSV file named test.csv into a pandas DataFrame named data. The encoding="latin1" is used to handle special characters in the CSV.
data['Q1'] = None, data['Q2'] = None, data['Q3'] = None: Adds three new columns (Q1, Q2, and Q3) to the DataFrame, initializing them with None. These columns will later store the top 3 recommendations.
url = "http://localhost:8000/query/": Sets the URL of the API endpoint where the script will send POST requests.

4. Function to Get Recommendations from the API
def get_recommendations(text):
    recommendations =["", "", ""]
    response = requests.post(url, json={"text":text})
    if response.status_code == 200:
        formatted_results = response.json()
        count = 0
        for result in formatted_results:
            recommendations[count] = f"**HS Code:** {result['HS Code']}  **Score:** {result['Score']}%"
            count += 1
            if count == 3:
                break
    return recommendations
get_recommendations(text): This function takes a text input (a product description) and queries the API to get HS code recommendations.
recommendations = ["", "", ""]: Initializes a list to store the top 3 recommendations, starting with empty strings.
response = requests.post(url, json={"text"
}): Sends a POST request to the API with the input text as JSON.
if response.status_code == 200: Checks if the request was successful (HTTP status code 200).
formatted_results = response.json(): Parses the JSON response into a Python dictionary.
for result in formatted_results: Loops through the results returned by the API.
recommendations[count] = f"HS Code: {result['HS Code']} Score: {result['Score']}%": Formats each recommendation and stores it in the recommendations list.
if count == 3: break: Limits the number of recommendations to 3.
return recommendations: Returns the list of recommendations.

5. Function to Process Each Row of the DataFrame
def process_row(row):
    index, data = row  
    text = data['Description']  
    recommendations = get_recommendations(text)
    return recommendations
process_row(row): This function processes a single row from the DataFrame.
index, data = row: Unpacks the tuple returned by df.iterrows() into index (row index) and data (the actual data in that row).
text = data['Description']: Extracts the Description field from the row, which contains the text to be processed.
recommendations = get_recommendations(text): Calls the get_recommendations function to get recommendations for the text.
return recommendations: Returns the list of recommendations for that row.

6. Applying Multithreading for Faster Processing
def apply_multiprocessing(df):
    num_threads = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_row, df.iterrows()), total=len(df)))

    for i, col_name in enumerate(['Q1', 'Q2', 'Q3']):
        df[col_name] = [result[i] for result in results]
    
    return df
apply_multiprocessing(df): This function applies multithreading to process the DataFrame faster.
num_threads = 4: Specifies the number of threads to use. Here, 4 threads are used to process rows concurrently.
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor: Creates a thread pool executor with the specified number of threads.
results = list(tqdm(executor.map(process_row, df.iterrows()), total=len(df))):
executor.map(process_row, df.iterrows()): Maps the process_row function to each row in the DataFrame using multithreading.
tqdm(...): Wraps the mapping process in a progress bar that shows the completion status.
list(...): Collects the results from the multithreaded execution into a list.
for i, col_name in enumerate(['Q1', 'Q2', 'Q3']): Loops over the column names Q1, Q2, Q3.
df[col_name] = [result[i] for result in results]: Fills each column (Q1, Q2, Q3) in the DataFrame with the corresponding recommendations from the results list.
return df: Returns the updated DataFrame with the recommendations.


7. Applying the Function and Saving Results
df_with_recommendations = apply_multiprocessing(data)
df_with_recommendations.to_csv("output.csv", index=False)
df_with_recommendations = apply_multiprocessing(data): Applies the multithreading function to the data DataFrame, generating recommendations for each row.
df_with_recommendations.to_csv("output.csv", index=False): Saves the updated DataFrame with recommendations to a CSV file named output.csv. The index=False argument ensures that row indices are not saved to the CSV file.

This script reads a CSV file, processes each row by sending its description to a FastAPI service for recommendations, and stores the top 3 recommendations in new columns (Q1, Q2, and Q3). The processing is done in parallel using multiple threads to speed up the operation. Finally, it saves the results to a new CSV file.


These is explanation for the code in notebook.py

1. Importing Required Libraries
import pandas as pd
import re
pandas as pd: A library for data manipulation and analysis, often used to handle tabular data.
re: A module that provides support for regular expressions, used for text processing.

2. Loading and Preparing the Data
df = pd.read_csv("htsdata.csv")
df = df[['HTS Number', 'Indent', 'Description']]

df.rename(columns={'HTS Number': 'HS CODE', 'Description': 'DESCRIPTION'}, inplace=True)
df = pd.read_csv("htsdata.csv"): Loads the CSV file named htsdata.csv into a pandas DataFrame named df.
df = df[['HTS Number', 'Indent', 'Description']]: Selects only the relevant columns: HTS Number, Indent, and Description.
df.rename(columns={'HTS Number': 'HS CODE', 'Description': 'DESCRIPTION'}, inplace=True): Renames the columns HTS Number to HS CODE and Description to DESCRIPTION for easier reference.

3. Filtering Data for Chapters
chapter = df[df['Indent'] == 0]
chapter = df[df['Indent'] == 0]: Filters the DataFrame to include only rows where the Indent column is 0, indicating that these rows represent chapter-level descriptions in the tariff schedule.

4. Importing Additional Libraries
import chromadb.utils.embedding_functions as embedding_functions
import json
import chromadb
from tqdm.auto import tqdm
chromadb.utils.embedding_functions: Provides functions for embedding text into vectors, which are useful for similarity searches.
json: A module for working with JSON data (JavaScript Object Notation), though it isn't explicitly used in the code you provided.
chromadb: A library for working with a Chroma database, which stores and retrieves vector embeddings.
tqdm.auto: A module that provides a progress bar, useful for visualizing the progress of loops.

5. Setting Up the Embedding Function and Chroma Client
ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
client = chromadb.PersistentClient()
ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl"): Initializes an embedding function using the "hkunlp/instructor-xl" model, which is used to convert text into vector embeddings.
client = chromadb.PersistentClient(): Creates a client for interacting with the Chroma database, which will store the embeddings.

6. Creating or Retrieving a Collection in ChromaDB
batch_size=200
collection1 = client.get_or_create_collection(name="chapter", metadata={"hnsw:space": "cosine"}, embedding_function= ef)
batch_size=200: Sets the batch size to 200, meaning 200 rows will be processed at a time.
collection1 = client.get_or_create_collection(name="chapter", metadata={"hnsw
": "cosine"}, embedding_function= ef): Creates or retrieves a collection named chapter in the Chroma database. This collection uses the cosine similarity metric and the specified embedding function (ef).

7. Inserting Chapter Data into the ChromaDB Collection
for i in tqdm(range(0, len(chapter), batch_size)):
    i_end = min(i+batch_size, len(chapter))
    collection1.upsert(
    documents=chapter['DESCRIPTION'].iloc[i:i_end].tolist(),
    metadatas=[row.to_dict() for _, row in chapter[i:i_end].iterrows()],
    ids=[str(x) for x in range(i, i_end)],
)
for i in tqdm(range(0, len(chapter), batch_size)): Loops through the chapter DataFrame in batches of size 200, with a progress bar to show the loop's progress.
i_end = min(i+batch_size, len(chapter)): Determines the end index for the current batch. This ensures that the loop doesn't go out of bounds if the last batch is smaller than 200.
collection1.upsert(...): Inserts or updates the documents (chapter descriptions) into the chapter collection in ChromaDB.
documents=chapter['DESCRIPTION'].iloc[i
].tolist(): Converts the chapter descriptions in the current batch to a list.
metadatas=[row.to_dict() for _, row in chapter[i
].iterrows()]: Converts each row in the batch to a dictionary, storing metadata associated with each document.
ids=[str(x) for x in range(i, i_end)]: Assigns unique IDs to each document in the batch.

8. Preprocessing Text
def preprocess_text(text):
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace("<il>", "").replace("</il>", "")
    text = re.sub(r'[;/:<>,()]', ' ', text)
    text = re.sub(r"\S*https?:\S*", '', text)
    text = re.sub("[''“”‘’…]", '', text)
    return text

df['DESCRIPTION'] = df['DESCRIPTION'].apply(preprocess_text)
preprocess_text(text): A function that preprocesses a given text by:
Converting all text to lowercase.
Removing specific HTML-like tags (<il> and </il>).
Replacing certain punctuation characters (;, /, :, etc.) with spaces.
Removing URLs.
Removing specific special characters like quotes.
df['DESCRIPTION'] = df['DESCRIPTION'].apply(preprocess_text): Applies the preprocess_text function to each entry in the DESCRIPTION column of the DataFrame.

9. Creating Concatenated Descriptions Based on Indentation
def create_concat_desc(row):
    indent_level = row['Indent']
    concat_desc = ''
    seen_indent = []
    for i in range(row.name, -1, -1):
        if df.loc[i, 'Indent'] in seen_indent:
          continue
        if df.loc[i, 'Indent'] == 0:
            break
        if df.loc[i, 'Indent'] <= indent_level:
            concat_desc = concat_desc + ' ' +  df.loc[i, 'DESCRIPTION']
            seen_indent.append(df.loc[i, 'Indent'])
    return concat_desc.strip()

df['Concatenated_Description'] = df.apply(create_concat_desc, axis=1)
create_concat_desc(row): A function that concatenates descriptions in a hierarchical manner based on the Indent level.
indent_level = row['Indent']: Gets the indent level of the current row.
concat_desc = '': Initializes an empty string to store the concatenated description.
for i in range(row.name, -1, -1): Loops backward from the current row to the beginning of the DataFrame.
if df.loc[i, 'Indent'] in seen_indent: continue: Skips indent levels that have already been processed.
if df.loc[i, 'Indent'] == 0: break: Stops if the chapter level (Indent = 0) is reached.
if df.loc[i, 'Indent'] <= indent_level: Adds the description to concat_desc if its indent level is less than or equal to the current row's level.
seen_indent.append(df.loc[i, 'Indent']): Records that this indent level has been processed.
return concat_desc.strip(): Returns the concatenated description, stripping any leading or trailing spaces.
df['Concatenated_Description'] = df.apply(create_concat_desc, axis=1): Applies the create_concat_desc function to each row in the DataFrame, creating a new column Concatenated_Description.

10. Dropping Duplicates and Filtering Data
df = df.drop_duplicates(subset=['Concatenated_Description'], keep='first')
df.dropna(inplace=True)
df = df[df['Indent'] != 0]
df['checker'] = df['HS CODE'].str[0:4]
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(subset=['Concatenated_Description'], keep='first'): Removes duplicate rows based on the Concatenated_Description column, keeping only the first occurrence.
df.dropna(inplace=True): Removes any rows that contain NaN (missing) values.
df = df[df['Indent'] != 0]: Filters out the rows where Indent is 0, as these represent chapter descriptions that are not needed anymore.
df['checker'] = df['HS CODE'].str[0:4]: Creates a new column checker that stores the first four characters of the HS CODE, which is likely used for matching or validation purposes.
df.reset_index(drop=True, inplace=True): Resets the index of the DataFrame, dropping the old index and creating a new one.


11. Inserting Processed Data into ChromaDB Collection
batch_size=200
collection = client.get_or_create_collection



1 -> py notebook.py
2 -> uvicorn app:app.py --reload
3 -> streamlit run semantic_serch_streamlit.py
4 -> npx localtunnel --port 8501