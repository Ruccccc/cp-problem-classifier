from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertPreTrainedModel, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn
import uvicorn
import json
import os

# === 自定義模型（支援 Hugging Face 格式）===
class DistilBertWithSoftLabel(DistilBertPreTrainedModel):
    def __init__(self, config, loss_weights=None):
        super().__init__(config)
        self.bert = DistilBertModel(config)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_weights = loss_weights if loss_weights is not None else torch.ones(config.num_labels)
        self.init_weights()  # 初始化權重

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        if labels is not None:
            loss_matrix = self.loss_fn(logits, labels)
            loss = (loss_matrix * self.loss_weights.to(logits.device)).mean()
            return {"logits": logits, "loss": loss}
        return {"logits": logits}

# === 載入 tokenizer、config、loss_weights、模型 ===
MODEL_DIR = "./best_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
config = DistilBertConfig.from_pretrained(MODEL_DIR)

loss_weights = torch.load(os.path.join(MODEL_DIR, "weights.pt"))
model = DistilBertWithSoftLabel.from_pretrained(MODEL_DIR, config=config, loss_weights=loss_weights)
model.eval()

# === 載入 label 名稱 ===
with open("./fast_api/labels.json", "r") as f:
    label_names = json.load(f)

# === FastAPI 設定 ===
app = FastAPI()
templates = Jinja2Templates(directory="./fast_api/templates")

class TextInput(BaseModel):
    text: str

# === JSON API ===
@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()

    threshold = 0.5
    predicted = [label_names[i] for i, p in enumerate(probs) if p >= threshold]
    return {"predicted_labels": predicted, "scores": probs}

# === 表單介面 ===
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, text: str = Form(...)):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()

    threshold = 0.5
    filtered = [(i, float(p)) for i, p in enumerate(probs) if p >= threshold]
    top = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
    predicted = [label_names[i] for i, _ in top]
    top5_probs = [probs[i] for i, _ in top]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "text": text,
            "predicted": predicted,
            "probs": top5_probs
        }
    })

# === 本地端啟動指令 ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
