FROM python:3.8.7
RUN pip install --upgrade pip
RUN apt update -y
RUN apt install build-essential cmake pkg-config -y
RUN useradd -ms /bin/bash worker
USER worker
WORKDIR /home/worker/app
COPY --chown=worker:worker requirements.txt README.md setup.py /home/worker/app/
RUN mkdir -p /home/worker/.local/bin
RUN mkdir -p /home/worker/liveness
ENV PATH="/home/worker/.local/bin:${PATH}"
RUN pip install --no-cache-dir --user -r requirements.txt
# Solo se debe utilizar una de las dos opciones a continuación
# Para descargar los modelos desde Internet
RUN python setup.py model_download
# En caso de tener descargados los modelos ubicados en la carpeta .deepface
# COPY --chown=worker:worker .deepface /home/worker/.deepface
LABEL maintainer="Daniel Jimenez <djimenezjerez@gmail.com>" version="1.0.0"
VOLUME ["/home/worker/app", "/home/worker/liveness"]
EXPOSE 5000
CMD ["python", "main.py"]