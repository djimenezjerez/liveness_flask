# MICROSERVICIO DE RECONOCIMIENTO FACIAL

## INSTALACIÓN EN UN ENTORNO LOCAL

* Verificar que la versión del entorno de ejecución Python sea igual a **3.8.7** mediante el comando:

```
$ python --version
```

* Actualizar el gestor de paquetes de Python

```
$ pip install --upgrade pip
```

* Instalar las dependencias

```
$ pip install -r requirements.txt
```

* Descargar los modelos pre-entrenados de aprendizaje de máquina

```
$ ./setup.py model_download
```

## DESPLIEGUE CON DOCKER

* Construir la imagen de python

```
$ docker build -t liveness_flask .
```

* Levantar el contenedor modificando los parámetros requeridos

```
$ docker run --name liveness_flask --restart=always -v ${PWD}:/home/worker/app -v DIRECTORIO_LARAVEL_PVT-BE/storage/app/liveness/faces:/home/worker/liveness -d -p 5000:5000 -d liveness_flask:latest
```

# LICENCIAS

- [GPLv3](LICENSE)
- [LPG-Bolivia](LICENCIA.txt)