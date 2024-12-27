#FROM python:3.7-slim-buster
##FROM google/cloud-sdk:slim@sha256:b18f4f31d806f49c97c5fa374a76716255b75a8d04dd133c3d3719b4753825d8
#WORKDIR /
#COPY requirements.txt /requirements.txt
#COPY src /src
#RUN pip install --upgrade pip && pip install -r requirements.txt
#ENTRYPOINT [ "bash" ]

#Docker file ORO

FROM python:3.10-slim
# ARG DOCKER_PROXY="dockerproxy.repos.tech.orange"
# FROM  ${DOCKER_PROXY}/python:3.10-slim

##ENV http_proxy=http://kirk.crm.orange.intra:3128
##ENV https_proxy=http://kirk.crm.orange.intra:3128
#ENV NO_PROXY="metadata.google.internal,bigquery.googleapis.com,*.googleapis.com"


#ENV http_proxy=http://172.19.93.163:3128
#ENV https_proxy=http://172.19.93.163:3128
ENV NO_PROXY="metadata.google.internal,bigquery.googleapis.com,*.googleapis.com"

COPY . /

COPY requirements.txt requirements.txt

RUN apt-get update -y && \
    apt-get upgrade -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

#RUN echo "172.19.93.163 kirk.crm.orange.intra" >> /etc/hosts

# Command to run your application
# CMD ["python", "fit_and_predict.py"]

#ENTRYPOINT ["python", "steps/train.py"]
#ENTRYPOINT ["gunicorn", "task.endpoint:app", "--bind=0.0.0.0:8080"] # TODO add threads and workers

#docker build . -t smart-capex-capacity-pipeline --build-arg http_proxy=http://kirk.crm.orange.intra:3128 --build-arg https_proxy=http://kirk.crm.orange.intra:3128 --build-arg NO_PROXY=metadata.google.internal,bigquery.googleapis.com,*.googleapis.com
