FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY /app/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY ./saved_models /saved_models/

COPY ./app /app
