# https://youtu.be/bluclMxiUkA
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 
Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.
"""


import numpy as np
from flask import Flask, request, render_template
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem.porter import *
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():

	input_email = request.form.values() #put into list to match our format 
	# tfidf_vectorizer = TfidfVectorizer()
	# tokenizer = tfidf_vectorizer.build_tokenizer()   
	# punct = string.punctuation
	# stemmer = PorterStemmer()

	# english_stops = set(stopwords.words('english'))

	# doc = input_email
	# doc = str(doc.lower())
	# doc = [i for i in doc if not (i in punct)] # non-punct characters
	# doc = ''.join(doc) # convert back to string
	# words = tokenizer(doc) # tokenizes
	# words = [w for w in words if w not in english_stops] #remove stop words
	# words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words] #stemmer and lemmatizer

	output = model.get_prediction()

	return render_template('index.html', prediction_text='Predicted Spam Class is {}'.format(output))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()
