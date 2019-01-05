#import
import os
import numpy as np
from flask import (Flask, request, jsonify, render_template, redirect, url_for, session, flash)

import pandas as pd

from text_utils import isNewsURL
from text_preprocessing import text_preprocessing
# models
from text_model import text_clf_model
from url_model import url_clf_model

#Server
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

@app.route('/')
def index():
  return render_template('index.html')


#----------------------------TEXT-------------------------
@app.route('/textclf', methods=['GET', 'POST'])
def textclf():
  if request.method == 'POST':
    # do stuff when the form is submitted
    text = request.form['textclf']
    
    textclf = text_clf_model(text)

    # redirect to end the POST handling
    # the redirect can be to the same route or somewhere else
    return render_template('textclf_results.html', result_list = textclf['results'], text = textclf['text'])

  # show the form, it wasn't submitted
  return render_template('textclf.html')

# @app.route('/textclf-results', methods= ['GET', 'POST'])
# def textclfResult():
#   pre_text = session['text']

#   return render_template('textclf_results.html', text=pre_text)


#----------------------------URL--------------------------
@app.route('/urlclf', methods=['GET', 'POST'])
def urlclf():
  if request.method == 'POST':
    # do stuff when the form is submitted
    url = request.form['urlclf'] 
    if(not isNewsURL(url)):
      error = "Bạn chưa nhập đúng định dạng URL, vui lòng nhập lại"
      return render_template('urlclf.html', error = error)
    try:
      urlclf = url_clf_model(url)
      return render_template('urlclf_results.html', result_list = urlclf['results'], url = urlclf['url'], result = urlclf)
    except Exception as e:
      return render_template('urlclf.html', error = e)
    # redirect to end the POST handling
    # the redirect can be to the same route or somewhere else
    # return redirect(url_for('textclf-results'))

  # show the form, it wasn't submitted
  return render_template('urlclf.html')


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


if __name__ == '__main__':
  app.run(debug = True)
  # Bind to PORT if defined, otherwise default to 5000.
  app.run()