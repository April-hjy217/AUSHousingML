FROM prefecthq/prefect:2-latest

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 4200

ENTRYPOINT ["prefect", "server", "start"]
CMD ["--host", "0.0.0.0", "--port", "4200"]

