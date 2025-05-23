FROM jupyter/minimal-notebook

COPY ./requirements-docker.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt