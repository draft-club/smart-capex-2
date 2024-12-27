FROM python:3.9.13-slim
WORKDIR /app

#ENV http_proxy=http://kirk.crm.orange.intra:3128
#ENV https_proxy=http://kirk.crm.orange.intra:3128

ARG country
ARG http_proxy
ARG https_proxy



COPY config ./config 
COPY src ./src 
COPY app.py ./app.py 
COPY requirements.txt requirements.txt
COPY schema_registry ./schema_registry
COPY dq_report ./dq_report
COPY logs ./logs

RUN case $country in\
         oci) mv schema_registry/oci/* config/;;\
         obf) mv schema_registry/obf/* config/;;\
         ordc) mv schema_registry/ordc/* config/;;\
         *) echo "country should be either obf, ordc or oci and not $country";;\
    esac


RUN pip install -r requirements.txt &&\
    export http_proxy=$http_proxy &&\
    export https_proxy=$https_proxy 
    

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD pip show numpy pandas


CMD [ "python3", "app.py"]
