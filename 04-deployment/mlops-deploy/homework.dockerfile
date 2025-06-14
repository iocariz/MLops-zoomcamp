FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    pip install --upgrade pip && \
    pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "batch/homework.py", "homework.py" ]

ENTRYPOINT [ "python", "homework.py" ]