import uvicorn
# FastAPI libray
from fastapi import FastAPI, UploadFile, File
from skipgan_inferrer import SkipganInferrer


# Initiate app instance
app = FastAPI(title='Screw Anomaly Detection', version='1.0',
              description='Skip-Ganomaly model is used for prediction')

skip_gan = SkipganInferrer()


@app.get('/')
def homepage():
    return {'Homepage is Oke'}

@app.post('/predict/')
def get_image(file: UploadFile = File(...)):
    result = skip_gan.infer(file.file)
    return {"Prediction": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)