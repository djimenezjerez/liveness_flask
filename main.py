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

@app.before_request
def only_json():
  if not request.is_json:
    return jsonify(message='Only json request is allowed'), 400

@app.route('/api/v1/analyze', methods=['POST'])
def get_analyze():
  request_data = request.get_json()
  if not 'image' in request_data:
    return jsonify(message='La cadena de texto en base64 image es requerida'), 400
  elif len(request_data['image']) < 100 or request_data['image'] is None:
    print(len(request_data['image']))
    return jsonify(message='La cadena de texto en base64 image debe tener al menos 100 caracteres'), 400
  else:
    contains_base64 = request_data['image'].find('base64,')
    if contains_base64 != -1:
      request_data['image'] = request_data['image'][contains_base64+7:]

  try:
    img = Image.open(BytesIO(base64.b64decode(request_data['image'])))
    img_file = '/tmp/'+''.join([random.choice(string.ascii_letters + string.digits) for n in range(30)])+'.jpg'
    img.save(img_file)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    faces = detector(img, 0)

    if len(faces) == 1:
      face3Dmodel = utils.ref3DModel()
      face = faces[0]
      shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
      refImgPts = utils.ref2dImagePoints(shape)
      height, width, channel = img.shape
      focalLength = 1 * width
      cameraMatrix = utils.cameraMatrix(focalLength, (height / 2, width / 2))
      mdists = np.zeros((4, 1), dtype=np.float64)

      # calculate rotation and translation vector using solvePnP
      success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)
      noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
      noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

      # calculating angle
      rmat, jac = cv2.Rodrigues(rotationVector)
      angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
      x = np.arctan2(Qx[2][1], Qx[2][2])
      y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
      z = np.arctan2(Qz[0][0], Qz[1][0])
      if angles[1] < -15:
        gaze = 'left'
      elif angles[1] > 15:
        gaze = 'right'
      else:
        gaze = 'forward'

      # DeepFace analysis
      df_analysis = DeepFace.analyze(img_path=img_file, actions=['age', 'gender', 'race', 'emotion'])
      os.remove(img_file)

      return jsonify({
        'message': 'Imagen analizada',
        'data': {
          'gaze': gaze,
          'analysis': df_analysis
        }
      })
  except:
    return jsonify({
      'message': 'Imagen inválida',
    }), 400


if __name__ == '__main__':
  app.run(
    app,
    host = '0.0.0.0',
    port = int(os.environ.get('APP_PORT')),
    debug = utils.str2bool(os.environ.get('APP_DEBUG')),
    threaded=False,
  )