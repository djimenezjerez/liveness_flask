#!/usr/bin/env python

from setuptools import setup, find_packages, command
from distutils import cmd, log
from os import path, mkdir
from pathlib import Path
from zipfile import ZipFile
import subprocess
import gdown
import bz2

def read(fname):
  return open(path.join(path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
  required_packages = f.read().splitlines()

class ModelDownloadCommand(cmd.Command):
  description = 'Descargar modelos pre-entrenados de detección facial de DLib'
  user_options = [
    ('model-download=', None, 'Descargar modelos pre-entrenados de detección facial de DLib'),
  ]
  home = str(Path.home())

  def initialize_options(self):
    self.model_download = ''

  def finalize_options(self):
    if self.model_download:
      assert path.exists(self.pylint_rcfile), (
        'Pylint config file %s does not exist.' % self.pylint_rcfile)

  def run(self):
    if not path.exists(self.home+"/.deepface"):
      mkdir(self.home+"/.deepface")
      print("Directory ",home,"/.deepface created")

    if not path.exists(self.home+"/.deepface/weights"):
      mkdir(self.home+"/.deepface/weights")
      print("Directory ",home,"/.deepface/weights created")

    if path.isfile(self.home+'/.deepface/weights/shape_predictor_68_face_landmarks.dat') != True:
      print("shape_predictor_68_face_landmarks.dat.bz2 is going to be downloaded")
      url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
      output = self.home+'/.deepface/weights/'+url.split("/")[-1]
      gdown.download(url, output, quiet=False)
      zipfile = bz2.BZ2File(output)
      data = zipfile.read()
      newfilepath = output[:-4]
      open(newfilepath, 'wb').write(data)

    if path.isfile(self.home+'/.deepface/weights/deploy.prototxt') != True:
      print("deploy.prototxt will be downloaded...")
      url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"
      output = self.home+'/.deepface/weights/deploy.prototxt'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel') != True:
      print("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")
      url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
      output = self.home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/shape_predictor_5_face_landmarks.dat') != True:
      print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")
      url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
      output = self.home+'/.deepface/weights/'+url.split("/")[-1]
      gdown.download(url, output, quiet=False)
      zipfile = bz2.BZ2File(output)
      data = zipfile.read()
      newfilepath = output[:-4]
      open(newfilepath, 'wb').write(data)

    if path.isfile(self.home+'/.deepface/weights/race_model_single_batch.h5') != True:
      print("race_model_single_batch.h5 will be downloaded...")
      url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
      output = self.home+'/.deepface/weights/race_model_single_batch.zip'
      gdown.download(url, output, quiet=False)
      with ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(self.home+'/.deepface/weights/')

    if path.isfile(self.home+'/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5') != True:
      print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")
      url = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
      output = self.home+'/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
      gdown.download(url, output, quiet=False)
      with ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(self.home+'/.deepface/weights/')

    if path.isfile(self.home+'/.deepface/weights/facial_expression_model_weights.h5') != True:
      print("facial_expression_model_weights.h5 will be downloaded...")
      url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
      output = self.home+'/.deepface/weights/facial_expression_model_weights.zip'
      gdown.download(url, output, quiet=False)
      with ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(self.home+'/.deepface/weights/')

    if path.isfile(self.home+'/.deepface/weights/gender_model_weights.h5') != True:
      print("gender_model_weights.h5 will be downloaded...")
      url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
      output = self.home+'/.deepface/weights/gender_model_weights.h5'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/deepid_keras_weights.h5') != True:
      print("deepid_keras_weights.h5 will be downloaded...")
      output = self.home+'/.deepface/weights/deepid_keras_weights.h5'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/age_model_weights.h5') != True:
      print("age_model_weights.h5 will be downloaded...")
      url = 'https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV'
      output = self.home+'/.deepface/weights/age_model_weights.h5'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/face-recognition-ensemble-model.txt') != True:
      print("face-recognition-ensemble-model.txt will be downloaded...")
      url = 'https://raw.githubusercontent.com/serengil/deepface/master/deepface/models/face-recognition-ensemble-model.txt'
      output = self.home+'/.deepface/weights/face-recognition-ensemble-model.txt'
      gdown.download(url, output, quiet=False)

    if path.isfile(self.home+'/.deepface/weights/dlib_face_recognition_resnet_model_v1.dat') != True:
      print("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")
      url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
      output = self.home+'/.deepface/weights/'+url.split("/")[-1]
      gdown.download(url, output, quiet=False)
      zipfile = bz2.BZ2File(output)
      data = zipfile.read()
      newfilepath = output[:-4]
      open(newfilepath, 'wb').write(data)

    if path.isfile(self.home+'/.deepface/weights/vgg_face_weights.h5') != True:
      print("vgg_face_weights.h5 will be downloaded...")
      gdown.download(url, self.home+'/.deepface/weights/vgg_face_weights.h5', quiet=False)

    self.announce('Modelos pre-entrenados descargados corréctamente', level=log.INFO)

setup(
  cmdclass = {
    'model_download': ModelDownloadCommand,
  },
  name = "Microservicio de reconocimiento facial",
  version = "1.0.0",
  author = "Daniel Jimenez",
  author_email = "djimenezjerez@gmail.com",
  description = ("Servicio de extracción de características faciales para realizar el control de vivencia y el reconocimiento facial"),
  license = "GPLv3",
  keywords = "liveness facial deepface python",
  url = "https://github.com/djimenezjerez/liveness_flask",
  install_requires = required_packages,
  packages = find_packages(),
  long_description = read('README.md'),
  classifiers = [
    "Development Status :: 1 - Production",
    "Topic :: Services",
    "License :: LPG-Bolivia :: GPLv3",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
  ],
  python_requires = '~=3.8.7',
)