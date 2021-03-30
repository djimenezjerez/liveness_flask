#!/usr/bin/env python

import utils
from dotenv import load_dotenv, find_dotenv
from sys import exit
if (find_dotenv()):
  load_dotenv(find_dotenv())
else:
  print('No se han definido las variables de entorno')
  exit(1)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_KEY')
app.config['TESTING'] = os.environ.get('FLASK_ENV') != 'production'
CORS(app)

import cv2
import dlib
import base64
import numpy as np
import string
import random
import re
from io import BytesIO
from pathlib import Path
from PIL import Image
from deepface import DeepFace

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(Path.home())+'/.deepface/weights/shape_predictor_68_face_landmarks.dat')

import uuid
import json
import face_recognition

# Construcción del modelo para acciones de análisis
available_actions = ['Emotion', 'Age', 'Gender', 'Race']
selected_actions = json.loads(os.environ.get('DF_ANALYZE_MODELS'))
actions = list(set(available_actions) & set(selected_actions))
if len(actions) == 0:
  print('Modelo de análisis desconocido, opciones diponibles: '+', '.join(available_actions))
  exit(1)
else:
  model_actions = {}
  for action in actions:
    model_actions[action.lower()] = DeepFace.build_model(action)

# Construcción de parámetros de análisis facial
mdists = np.zeros((4, 1), dtype=np.float64)

# Restringir consultas a application/json
@app.before_request
def only_json():
  if not request.is_json:
    return jsonify(message='Solo se permiten consultas de tipo application/json'), 400

# Generar modelo facial en base a una imagen
@app.route('/api/v1/build', methods=['POST'])
def build_post():
  request_data = request.get_json()

  # Validación parámetro image
  if not 'image' in request_data:
    return jsonify(message='El parámetro image es requerido'), 400
  else:
    if not isinstance(request_data['image'], str) or len(request_data['image']) < 5 or request_data['image'] is None:
      return jsonify(message='El parámetro image es requerido'), 400

  # Validación parámetro path
  if not 'path' in request_data:
    return jsonify(message='El parámetro path es requerido'), 400
  elif not isinstance(request_data['path'], str) or request_data['path'] is None:
    return jsonify(message='El parámetro path debe ser una cadena de texto'), 400
  else:
    if not os.path.exists(request_data['path']):
      return jsonify(message='Ruta de almacenamiento es inválida'), 400

  image_path = os.path.join(request_data['path'], request_data['image'])
  if not os.path.exists(image_path):
    return jsonify(message='Archivo inexistente: '+image_path), 400

  try:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if len(encoding) != 1:
      return jsonify({
        'message': 'Imagen inválida',
      }), 400
    encoding_file = os.path.join(request_data['path'], request_data['image'].split('.')[0]+'.npy')
    np.save(encoding_file, encoding[0])
    return jsonify({
      'message': 'Modelo generado',
      'data': {
        'file': encoding_file
      }
    })
  except:
    return jsonify({
      'message': 'Imagen inválida',
    }), 400

# Eliminar imagen y modelo facial
@app.route('/api/v1/remove', methods=['POST'])
def remove_post():
  request_data = request.get_json()

  # Validación parámetro image
  if not 'image' in request_data:
    return jsonify(message='El parámetro image es requerido'), 400
  else:
    if not isinstance(request_data['image'], str) or len(request_data['image']) < 5 or request_data['image'] is None:
      return jsonify(message='El parámetro image es requerido'), 400

  # Validación parámetro path
  if not 'path' in request_data:
    return jsonify(message='El parámetro path es requerido'), 400
  elif not isinstance(request_data['path'], str) or request_data['path'] is None:
    return jsonify(message='El parámetro path debe ser una cadena de texto'), 400
  else:
    if not os.path.exists(request_data['path']):
      return jsonify(message='Ruta de almacenamiento es inválida'), 400

  image_path = os.path.join(request_data['path'], request_data['image'])
  if os.path.exists(image_path):
    os.remove(image_path)

  encoding_file = os.path.join(request_data['path'], request_data['image'].split('.')[0]+'.npy')
  if os.path.exists(encoding_file):
    os.remove(encoding_file)

  return jsonify({
    'message': 'Archivos eliminados',
    'data': {
      'deleted': {
        'image': image_path,
        'model': encoding_file
      }
    }
  })

# Comparar una imagen con los modelos generados en un directorio
@app.route('/api/v1/verify', methods=['POST'])
def verify_post():
  request_data = request.get_json()

  # Validación parámetro threshold
  if not 'threshold' in request_data:
    threshold = 0.4
  else:
    if not isinstance(request_data['threshold'], float) or request_data['threshold'] <= 0 or request_data['threshold'] is None:
      return jsonify(message='El parámetro threshold debe ser mayor a 0'), 400
    else:
      threshold = request_data['threshold']

  # Validación parámetro image
  if not 'image' in request_data:
    return jsonify(message='El parámetro image es requerido'), 400
  else:
    if not isinstance(request_data['image'], str) or len(request_data['image']) < 5 or request_data['image'] is None:
      return jsonify(message='El parámetro image es requerido'), 400

  # Validación parámetro path
  if not 'path' in request_data:
    return jsonify(message='El parámetro path es requerido'), 400
  elif not isinstance(request_data['path'], str) or request_data['path'] is None:
    return jsonify(message='El parámetro path debe ser una cadena de texto'), 400
  else:
    if not os.path.exists(request_data['path']):
      return jsonify(message='Ruta de almacenamiento es inválida'), 400

  image_path = os.path.join(request_data['path'], request_data['image'])
  if not os.path.exists(image_path):
    return jsonify(message='Archivo inexistente: '+image_path), 400

  try:
    files = [np.load(os.path.join(request_data['path'], f)) for f in os.listdir(request_data['path']) if f.endswith('.npy')]
    if len(files) < 1:
      return jsonify(message='No existen suficientes modelos para comparar'), 400

    test_image = face_recognition.load_image_file(image_path)
    test_encoding = face_recognition.face_encodings(test_image)
    if len(test_encoding) != 1:
      return jsonify({
        'message': 'Imagen inválida',
      }), 400

    distances = face_recognition.face_distance(files, test_encoding[0])
    if np.mean(distances) <= threshold:
      verified = True
    else:
      verified = False

    return jsonify({
      'message': 'Verificación realizada',
      'data': {
        'verfied': verified,
        'distances': distances.tolist()
      }
    })
  except:
    return jsonify({
      'message': 'Imagen inválida',
    }), 400

# Comparar una imagen con los modelos generados en un directorio
@app.route('/api/v1/analyze', methods=['POST'])
def analyze_post():
  request_data = request.get_json()

  # Validación parámetro path
  if not 'path' in request_data:
    return jsonify(message='El parámetro path es requerido'), 400
  elif not isinstance(request_data['path'], str) or request_data['path'] is None:
    return jsonify(message='El parámetro path debe ser una cadena de texto'), 400
  else:
    if not os.path.exists(request_data['path']):
      return jsonify(message='Ruta de almacenamiento es inválida'), 400
    else:
      image_path = request_data['path']

  # Validación parámetro is_base64
  if not 'is_base64' in request_data:
    return jsonify(message='El parámetro is_base64 es requerido'), 400
  elif not isinstance(request_data['is_base64'], str) and not isinstance(request_data['is_base64'], bool) and not isinstance(request_data['is_base64'], int) or request_data['is_base64'] is None:
    return jsonify(message='El parámetro is_base64 debe ser booleano'), 400
  else:
    if isinstance(request_data['is_base64'], int):
      if request_data['is_base64'] not in (1, 0):
        return jsonify(message='El parámetro is_base64 debe ser booleano'), 400
      else:
        is_base64 = bool(request_data['is_base64'])
    elif isinstance(request_data['is_base64'], bool):
      is_base64 = request_data['is_base64']
    elif isinstance(request_data['is_base64'], str):
      is_base64 = utils.str2bool(request_data['is_base64'])

  # Validación parámetro image
  if not 'image' in request_data:
    return jsonify(message='El parámetro image es requerido'), 400
  else:
    if is_base64:
      if not isinstance(request_data['image'], str) or len(request_data['image']) < 100 or request_data['image'] is None:
        return jsonify(message='La cadena de texto en base64 image debe tener al menos 100 caracteres'), 400
      else:
        image_path = os.path.join(image_path, str(uuid.uuid4().hex)+'.jpg')
    else:
      image_path = os.path.join(image_path, request_data['image'])
      if not os.path.exists(image_path):
        return jsonify(message='Archivo inexistente: '+image_path), 400

  try:
    if is_base64:
      contains_base64 = request_data['image'].find('base64,')
      if contains_base64 != -1:
        request_data['image'] = request_data['image'][contains_base64+7:]
      img = Image.open(BytesIO(base64.b64decode(request_data['image'])))
      img.save(image_path)
      img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    else:
      img = cv2.imread(image_path, 1)

    faces = detector(img, 0)

    if len(faces) == 1:
      face3Dmodel = utils.ref3DModel()
      face = faces[0]
      shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
      refImgPts = utils.ref2dImagePoints(shape)
      height, width, channel = img.shape
      focalLength = 1 * width
      cameraMatrix = utils.cameraMatrix(focalLength, (height / 2, width / 2))

      # Cálculo de vector de rotación y traslación mediante solvePnP
      success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)

      # Cálculo del ángulo del rostro
      rmat, jac = cv2.Rodrigues(rotationVector)
      angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
      if angles[1] < -15:
        gaze = 'left'
      elif angles[1] > 15:
        gaze = 'right'
      else:
        gaze = 'forward'

      # DeepFace analysis
      df_analysis = DeepFace.analyze(image_path, models=model_actions, actions=selected_actions, detector_backend=os.environ.get('DF_ANALYZE_BACKEND'))

      return jsonify({
        'message': 'Imagen analizada',
        'data': {
          'file': image_path,
          'gaze': gaze,
          'analysis': df_analysis
        }
      })
  except:
    return jsonify({
      'message': 'Imagen inválida',
    }), 400

# Aplicación principal
if __name__ == '__main__':
  app.run(
    app,
    host='0.0.0.0',
    port=int(os.environ.get('APP_PORT')),
    debug=utils.str2bool(os.environ.get('APP_DEBUG')),
    threaded=False,
  )