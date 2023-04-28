# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

### models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
filename = 'dataset_with_emb.csv'

### get dataset
df = pd.read_csv(filename)
df['embedding'] = df['embedding'].apply(ast.literal_eval)
# print(df)

### Calculate get emb cost
def num_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))



### search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    ###Returns a list of strings and relatednesses, sorted from most related to least.###
    ### 從官方MODEL獲取發問內容的embedding數值###
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    ###整合dataset的embedding和發問內容的embedding, 選出符合度較高者###
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]



### Ask
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=5)
    ###  打印前5相關資料
    # for string, relatedness in zip(strings, relatednesses):
    #     print(f"{relatedness=:.3f}")
    #     print(f"{string}")

    ### token 127
    ### 你是GovitaTechLimited公司的客戶服務機械人，根據以下括號內的參考資料回答的問題，如果答案在資料中找不到，則回答"沒有相關訊息",以中文回答
    ### token 50
    ### You are a customer service chatbot for GovitaTech Limited. Please answer the following questions based on the reference materials provided in parentheses. If the answer cannot be found in the materials, please respond with "No relevant information found.Answer in english.
    
    question = f"\n\nQuestion: {query}"
    message = ""
    for string in strings:
        next_article = f"\n{string}"
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def requestGPT(prompt):
    introduction = """你是GovitaTechLimited公司的客戶服務機械人，只回答客服相關的對話，根據提供的參考資料回答問題，如果答案在資料中找不到，則回答"沒有相關訊息",以中文回答,回答格式為{答案內容}"""
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages = [
            {"role": "system", "content":introduction},
            {"role": "user", "content": prompt},
            ]
    )	
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = query_message("泰乐产品规格？", df, GPT_MODEL, token_budget=1000)
    response = requestGPT(prompt)
    print(response)

