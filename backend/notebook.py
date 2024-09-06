import pandas as pd
import re

df = pd.read_csv("htsdata.csv")
df = df[['HTS Number', 'Indent', 'Description']]

df.rename(columns={'HTS Number': 'HS CODE', 'Description': 'DESCRIPTION'}, inplace=True)

chapter = df[df['Indent'] == 0]

from chromadb.utils.embedding_functions.instructor_embedding_function import InstructorEmbeddingFunction
import json
import chromadb
from tqdm.auto import tqdm

ef = InstructorEmbeddingFunction(
model_name="hkunlp/instructor-xl")

client = chromadb.PersistentClient()
# client.delete_collection(name="chapter")
# client.delete_collection(name="hscode")

batch_size=200
collection1 = client.get_or_create_collection(name="chapter", metadata={"hnsw:space": "cosine"}, embedding_function= ef)

for i in tqdm(range(0, len(chapter), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(chapter))
    collection1.upsert(
    documents=chapter['DESCRIPTION'].iloc[i:i_end].tolist(),
    metadatas=[row.to_dict() for _, row in chapter[i:i_end].iterrows()],
    ids=[str(x) for x in range(i, i_end)],
)

def preprocess_text(text):
    text = " ".join([x.lower() for x in text.split()])
    text = text.replace("<il>", "").replace("</il>", "")
    text = re.sub(r'[;/:<>,()]', ' ', text)
    text = re.sub(r"\S*https?:\S*", '', text)
    text = re.sub("[''“”‘’…]", '', text)
    return text

df['DESCRIPTION'] = df['DESCRIPTION'].apply(preprocess_text)

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


df = df.drop_duplicates(subset=['Concatenated_Description'], keep = 'first')

df.dropna(inplace = True)

df = df[df['Indent'] != 0]

df['checker'] = df['HS CODE'].str[0:4]

df.reset_index(drop= True, inplace=True)

batch_size=200
collection = client.get_or_create_collection(name="hscode", metadata={"hnsw:space": "cosine"}, embedding_function= ef)

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    collection.upsert(
    documents=df['Concatenated_Description'].iloc[i:i_end].tolist(),
    metadatas=[row.to_dict() for _, row in df[i:i_end].iterrows()],
    ids=[str(x) for x in range(i, i_end)],
)
  