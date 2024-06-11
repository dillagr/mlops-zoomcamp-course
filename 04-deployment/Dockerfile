FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim as builder-image
WORKDIR /app
ENV PATH=$PATH:/app

## copy files
COPY ["main.py", "Pipfile", "Pipfile.lock", "./"]
RUN pip install --upgrade pip pipenv
RUN pipenv install --system

EXPOSE 8080

## make sure serve is executable
# RUN chmod +x serve
ENTRYPOINT [ "gunicorn", "-b", "0.0.0.0:8080", "main:app" ]

