FROM python:latest
RUN pip install --upgrade pip
WORKDIR /app
COPY flask_main.py .
EXPOSE 8100
RUN pip install -r requirements.txt
CMD ["python", "flask_main.py"]