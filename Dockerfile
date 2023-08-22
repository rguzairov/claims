# FROM jupyter/minimal-notebook

# USER root

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc

# RUN conda install -c conda-forge wordcloud
# RUN conda install -c conda-forge ipywidgets

# COPY requirements.txt .
# RUN pip install -r requirements.txt

FROM tiangolo/uwsgi-nginx-flask:python3.10

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application into image
COPY . /app

ENV \
    STATIC_PATH=/app/app/static \
    FLASK_ENV=production

EXPOSE 80