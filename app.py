from fastapi import FastAPI
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline, TextClassificationPipeline
import os
import torch
import gdown

'''
heroku url : https://lit-spire-48980.herokuapp.com/
'''
app = FastAPI()


@app.get("/{input}")
def predict(input: str):
    model_filename = 'model_distilbert.bin'
    tokenizer_filename = 'tokenizer_distilbert.bin'
    if not os.path.isfile(model_filename):
        tokenizer_url = 'https://drive.google.com/uc?id=1O46AxalGC7-Ck8r2Z2196Z_qN5DWBCOX'
        model_url = 'https://drive.google.com/uc?id=14PHQB2SNKOgn0dW1E6q-raOx-lAJnRgr'
        gdown.download(model_url, model_filename)
        gdown.download(tokenizer_url, tokenizer_filename)
        # model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
        # tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

        # with open(tokenizer_filename, 'wb') as f:
        #     pickle.dump(tokenizer, f)
        # with open(model_filename, 'wb') as f:
            # pickle.dump(model, f)           
        # torch.save(model, model_filename)
        # torch.save(tokenizer, tokenizer_filename)
    model = torch.load(model_filename)
    tokenizer = torch.load(tokenizer_filename)
    # with open(tokenizer_filename, 'rb') as f:
    #     tokenizer = pickle.load(f)
    # with open(model_filename, 'rb') as f:
    #     model = pickle.load(f)
    distilled_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    x = distilled_classifier(input)[0]
    x = {item['label']:item['score'] for item in x}
    predictions = dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
    return {'text': input, 'predictions': predictions}



# md5 = 'fa837a88f0c40c513d975104edf3da17'
# gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)


    
