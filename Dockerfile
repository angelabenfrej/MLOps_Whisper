FROM python:3.10-slim-bullseye

RUN apt update -y && pip install --upgrade pip

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt   --retries=10

CMD ["python3", "app.py"]