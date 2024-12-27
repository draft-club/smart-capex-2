
FROM python:3.7-slim-buster
WORKDIR /
## Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src /src
## Sets up the entry point to invoke the trainer
ENTRYPOINT ["python", "-m", "src.main"]
