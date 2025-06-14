FROM python:3.9.6-slim  

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["web-service/predict.py", "./"]
RUN mkdir -p models
COPY ["web-service/models/lin_reg.bin", "./models/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"] 