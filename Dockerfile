# syntax=docker/dockerfile:experimental
FROM python:3.8-buster

WORKDIR /app
COPY . /app

# patch security packages
RUN apt-get update; apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh,id=github_ssh_key pip install --upgrade pip setuptools && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords && \
    python -m nltk.downloader punkt
CMD ["uvicorn", "de_server:app", "--workers", "8", "--host", "0.0.0.0", "--port", "5000"]
