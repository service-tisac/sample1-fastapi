from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
from textblob import TextBlob
import emoji
import time

class TextRequest(BaseModel):
    text: str

class InputData(BaseModel):
    input: List[float]

# Define the simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the GPU-intensive Neural Network
class IntensiveNN(nn.Module):
    def __init__(self):
        super(IntensiveNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.layers(x)

# Load model from checkpoint
def load_model(checkpoint_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Create FastAPI app
app = FastAPI()

# Load the simple model checkpoint
simple_model = load_model('simple_nn_checkpoint.pth')

# Initialize the intensive model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intensive_model = IntensiveNN().to(device)

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
        output = simple_model(input_tensor)
    
    return {"output": output.squeeze().item()}

@app.get("/test")
async def test_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return {"gpu_available": True, "gpu_name": gpu_name}
    else:
        return {"gpu_available": False}

@app.post("/gpu_intensive")
async def gpu_intensive():
    start_time = time.time()
    
    # Generate a large random input
    input_size = 10000
    input_tensor = torch.randn(input_size, 512, device=device)
    
    # Perform forward and backward passes multiple times
    num_iterations = 100
    optimizer = torch.optim.Adam(intensive_model.parameters(), lr=0.001)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = intensive_model(input_tensor)
        loss = output.sum()
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    memory_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    memory_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    
    return {
        "message": "GPU-intensive task completed",
        "execution_time": execution_time,
        "gpu_memory_allocated_gb": memory_allocated,
        "gpu_memory_reserved_gb": memory_reserved
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
