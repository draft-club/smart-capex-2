FROM python:3.9.13-slim
RUN useradd -m myuser
USER myuser

WORKDIR /app

COPY src ./src
COPY data/samples ./data/samples
COPY config ./config
COPY .env .env
COPY main_randim.py ./main_randim.py
COPY requirements.txt requirements.txt
COPY running_config.ini running_config.ini
RUN mkdir "logs"

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/src"

CMD python3 src/d06_api/api.py --path_to_country_parameters ${PATH_FILE_COUNTRY_PARAMETER_ENV}
