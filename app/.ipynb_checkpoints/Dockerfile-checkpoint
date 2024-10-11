# FROM python:3.9-slim@sha256:980b778550c0d938574f1b556362b27601ea5c620130a572feb63ac1df03eda5 
FROM python:3.11.8-slim

WORKDIR /app

COPY [ ".", "./" ]

RUN pip install -r requirements.txt
RUN pip install google-cloud-storage

EXPOSE 8080
# As an example here we're running the web service with one worker on uvicorn.
# CMD exec uvicorn main:app --host 0.0.0.0 --port 1234 --workers 1

ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

