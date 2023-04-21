
##################
#### DEFINE  #####
##################

# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# an example question about the 2022 Olympics
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response['choices'][0]['message']['content'])


####################################
#### Prepare search data  ##########
####################################

## Embedding Wikipedia articles for search.
## https://github.com/openai/openai-cookbook/blob/d67c4181abe9dfd871d382930bb778b7014edc66/examples/Embedding_Wikipedia_articles_for_search.ipynb

embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"

df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)
# the dataframe has two columns: "text" and "embedding"
#                       text	                                            embedding
# 0	Lviv bid for the 2022 Winter Olympics\n\n{{Oly...	[-0.005021067801862955, 0.00026050032465718687...
# 1	Lviv bid for the 2022 Winter Olympics\n\n==His...	[0.0033927420154213905, -0.007447326090186834,...


####################################
############## Search  #############
####################################

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    """ 從官方MODEL獲取發問內容的embedding數值"""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    """整合dataset的embedding和發問內容的embedding, 選出符合度較高者"""
    strings_and_relatednesses = [
                        """計算問題 emb與dataset emb 的差"""
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
# examples
strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    #relatedness=0.879
    #'Curling at the 2022 Winter Olympics\n\n==Medal summary==\n\n===Medal table===\n\n{{Medals table\n | caption        = \n | host           = \n | flag_template  = flagIOC\n | event          = 2022 Winter\n | team           = \n | gold_CAN = 0 | silver_CAN = 0 | bronze_CAN = 1\n | gold_ITA = 1 | silver_ITA = 0 | bronze_ITA = 0\n | gold_NOR = 0 | silver_NOR = 1 | bronze_NOR = 0\n | gold_SWE = 1 | silver_SWE = 0 | bronze_SWE = 2\n | gold_GBR = 1 | silver_GBR = 1 | bronze_GBR = 0\n | gold_JPN = 0 | silver_JPN = 1 | bronze_JPN - 0\n}}'



####################################
############## Ask   ###############
####################################
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string. """
    """ TOKEN 計算"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


""" 等於Prompt , """
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    """ 獲取與對應的dataset答案，已計算最高匹配度，包括text與相關值並排序"""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    """ 設定背景 , 整合所有發問內容(prompt)"""
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction

    """ 計算每段內容token有無超出budget限制,並累加在message中,  只保留範圍以內的內容 """
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


"""  將整理好的相關內容(dataset)與發問內容一同發送給GPT作思考"""
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message