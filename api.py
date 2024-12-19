from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from gensim.models import KeyedVectors
import MeCab


app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 任意のオリジンを許可（必要に応じて設定を変更）
    allow_credentials=True,
    allow_methods=["*"],  # 任意のメソッドを許可（必要に応じて設定を変更）
    allow_headers=["*"],  # 任意のヘッダーを許可（必要に応じて設定を変更）
)

wv = KeyedVectors.load_word2vec_format('wiki.model', binary=True)

@app.post("/word-embedding")
async def get_word_embedding(data: dict = Body(...)):
    try:
        positive = data["positive"]
        negative = data["negative"]
        
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise HTTPException(status_code=400, detail="Both positive and negative should be lists of words.")
        
        results = wv.most_similar(positive=positive, negative=negative)
        
        return {"results": results}
    except KeyError:
        raise HTTPException(status_code=400, detail="Both positive and negative keys are required in the JSON.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
