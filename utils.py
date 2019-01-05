import ast, os
from sklearn.externals import joblib

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

"""
load_n_gram(corpus_path): load data from corpus_path
"""
def load_n_gram(corpus_path):
  with open(corpus_path, encoding = 'utf-8') as file:
    words = file.read()
    words = ast.literal_eval(words)
    return words

"""
load_file_with_newline: load words from '<file_path>.txt' file separate with \n notation
Output: Set of words
Stopwords.txt description: 1 word 1 line + 2 python style commments
"""
def load_file_with_newline(file_path):
  words = set()
  with open(file_path, encoding='utf-8') as file:
    for line in file:
      li=line.strip()
      if not li.startswith("#"):
        words.add(line.rstrip())
              
  return words

"""
load_model: load model from model path
"""
def load_model(model_path, model_file): 
  return joblib.load(os.path.join(BASE_DIR, model_path, model_file))
