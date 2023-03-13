import sys

# Model Imports
from models.TFIDF import *
import pickle

print("Baseline TF-IDF model:")
class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']

object_token = tfidf_Tokenization(class_lis, 'data')
token = object_token.token_X() #data_path in the model file
seeds = object_token.modify_seeds()

object_tfidf = tfidf(token, seeds, class_lis, 'data')

#save model
print("saveing model pickle file")
pickle.dump(object_tfidf, open('models/model_tfidf.pkl', 'wb'))

