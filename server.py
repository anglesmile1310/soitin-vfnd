#import
import os
import numpy as np
from flask import Flask, request, jsonify

import pandas as pd

#Server
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

@app.route('/')
def index():
  return 'Index Page'

@app.route('/textclf')
def textClf():
  return 'textClf'

@app.route('/urlclf')
def urlClf():
  return 'URL clf'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % subpath