import openai
import pandas as pd
import csv
import tiktoken


filename = "JD_FAQ"
df = pd.read_csv(f'{filename}_embformat.csv')
### 将两列组合为一个字符串列### 
df['combined'] = df.apply(lambda x: str(x['question']) + str(x['answer']), axis=1)
dataset_str = df['combined'].tolist()

EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request


### Calculate get emb cost
def num_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_cost(dataset_str, model):
    dataset_str_all = "".join(dataset_str)
    total_tokens = num_tokens(dataset_str_all, model)
    model_cost = {"gpt-3.5-turbo":0.002, "text-embedding-ada-002":0.0004}
    final_cost = model_cost[model]*total_tokens/ 1000
    return final_cost

print(count_cost(dataset_str,EMBEDDING_MODEL))

### Create embeddings 
embeddings = []
for batch_start in range(0, len(dataset_str), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = dataset_str[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

### Save 
df = pd.DataFrame({"text": dataset_str, "embedding": embeddings})
SAVE_PATH = "dataset_with_emb.csv"
df.to_csv(SAVE_PATH, index=False)


"""Because this example only uses a few thousand strings, we'll store them in a CSV file."""
"""For larger datasets, use a vector database, which will be more performant."""

