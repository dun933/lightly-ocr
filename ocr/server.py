import logging
import os

import flask
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from pipeline import serveModel

UPLOAD_FOLDER = f'{os.path.dirname(os.path.relpath(__file__))}/test'
ALLOWED = {'png', 'jpeg', 'jpg'}

app = Flask(__name__)
app.config.from_mapping(SECRET_KEY='development')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def isAllowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/', methods=['GET'])
def isOnline():
    logging.info('ping received')
    return jsonify({'status': 'online'}), 200


@app.route('/api', methods=['POST'])
def parseText():
    if 'file' not in request.file:
        logging.warn('does not receive an image')
        return jsonify({'status': 'noInput'}), 403
    file = request.file['file']
    if file.filename == '':
        logging.warn('no input received')
        return jsonify({'status': 'emptyInput'}), 403
    if file and isAllowed(file.filename):
        filename = secure_filename(file.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fpath)
        logging.info(f'got an inputs, saved at {fpath}')
        results = m.predict(fpath)
        return jsonify({'status': 'OK', 'results': {k: v for k, v in enumerate(results)}}), 200


if __name__ == '__main__':
    logging.info('Starting server...')
    m = serveModel('config.yml', thresh=0.6)
    app.run(host='0.0.0.0', port=5000)
