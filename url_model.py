from sklearn.externals import joblib
import os
from newsplease import NewsPlease
import unicodedata as ud

from utils import BASE_DIR, load_model
from get_keywords import get_topn_keywords_model


from text_preprocessing import text_preprocessing

#path
model_path = 'models'

text_models = [
    'MultiNB_clf_1.sav',
    'MultiNB_clf_2.sav',
    'MultiNB_clf_textdomain_2.sav'
]


'''predict_text: predict text by model.
      @Output: Dictionary of classify result
'''
def predict_text(model, text, domain):
  result_dict = dict()
  #name
  result_dict['name'] = model['name']

  if('textdomain' in model['name']):
    #probability
    proba = model['model'].predict_proba([text + domain]).tolist()[0]
    result_dict['proba'] = [float(i)/sum(proba) for i in proba]

     #label
    result_dict['label'] = model['model'].predict([text + domain]).tolist()[0]
    
    #keywords
    result_dict['keywords'] = get_topn_keywords_model(text + domain, model['model'], 10)

  else:
    #probability
    proba = model['model'].predict_proba([text]).tolist()[0]
    result_dict['proba'] = [float(i)/sum(proba) for i in proba]

    #label
    result_dict['label'] = model['model'].predict([text]).tolist()[0]
    
    #keywords
    result_dict['keywords'] = get_topn_keywords_model(text, model['model'], 10)

  return result_dict

def url_clf_model(urltext):
  url_clf_result = dict()
  news = NewsPlease.from_url(urltext).__dict__

  if (news.get('text') == None):
    raise ValueError("Không có dữ liệu trong văn bản của tin tức")
  elif(news.get("title") == None):
    raise ValueError("Không có dữ liệu trong tiêu đề của tin tức")
  else:
    
    models = [{'name': model.split('.')[0] , 'model': load_model(model_path,model)} for model in text_models]
    
    # url_clf_result is return value
    url_clf_result['url'] = urltext
    url_clf_result['title'] = news['title']
    url_clf_result['time'] = news['date_publish']
    url_clf_result['domain'] = news['source_domain']
    url_clf_result['author'] = news['authors']

    preproc_text = text_preprocessing(news['text'])
    url_clf_result['results'] = [predict_text(model, preproc_text, url_clf_result['domain']) for model in models]
    print(url_clf_result)
  return url_clf_result
