import os
import unicodedata as ud

from utils import BASE_DIR, load_model
from text_preprocessing import text_preprocessing
from get_keywords import get_topn_keywords_model
#Path
model_path = 'models'

text_models = [
    'MultiNB_clf_1.sav',
    'MultiNB_clf_2.sav'
]

'''predict_text: predict text by model.

'''
def predict_text(model, text):
  result_dict = dict()
  #name
  result_dict['name'] = model['name']

  
  #probability
  proba = model['model'].predict_proba([text]).tolist()[0]
  result_dict['proba'] = [float(i)/sum(proba) for i in proba]

  #label
  result_dict['label'] = model['model'].predict([text]).tolist()[0]
  
  #keywords
  result_dict['keywords'] = get_topn_keywords_model(text, model['model'], 10)

  return result_dict


''''text_clf_model: main model to predict.
      @Input: text 
      @Output: dictionary of result
          {
            'text': text content,
            'results': [list of result: {
                      'name': model name,
                      'proba': probability result of prediction,
                      'label': label of prediction
                      'keywords': list of key word
                }],
          }
'''
def text_clf_model(text):
  text_clf_result = dict()
  text_clf_result['text'] = ud.normalize("NFC", text)

  preproc_text = text_preprocessing(text_clf_result['text'])

  models = [{'name': model.split('.')[0] , 'model': load_model(model_path,model)} for model in text_models]

  
  text_clf_result['results'] = [predict_text(model, preproc_text) for model in models]

  print(text_clf_result)
  return text_clf_result


  
