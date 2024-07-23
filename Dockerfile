FROM python:3.10-slim-bullseye

RUN apt update -y 
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]