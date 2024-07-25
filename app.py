from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
from textblob import TextBlob
import emoji

class TextRequest(BaseModel):
    text: str

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model from checkpoint
def load_model(checkpoint_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Define input data model
class InputData(BaseModel):
    input: List[float]

# Create FastAPI app
app = FastAPI()

# Load the model checkpoint
model = load_model('simple_nn_checkpoint.pth')

@app.post("/test/textblob")
async def test_textblob(request: TextRequest):
    try:
        blob = TextBlob(request.text)
        sentiment = blob.sentiment
        response = {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/emoji")
async def test_emoji():
    try:
        response = {
            "message": emoji.emojize("Hello, world! :earth_americas:")
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: InputData):
    if len(data.input) != 10:
        raise HTTPException(status_code=400, detail="Input must be a list of 10 floats.")
    
    input_tensor = torch.tensor(data.input).float().unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    
    return {"output": output.squeeze().item()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
