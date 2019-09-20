# import libraries
from sqlalchemy import create_engine
import pandas, string, re, argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from nltk.stem.wordnet import WordNetLemmatizer
import pickle




def make_predictions(classifier, X_test_set, Y_test_set):
    """
    This function automates the output format for displaying the results for a trained model, classifier using the 
    """
    predictions  = classifier.predict(X_test_set)
    predictions = pandas.DataFrame(predictions, columns= Y_test_set.columns)
    for y in Y_test_set.columns:
        print("\n\n" +y + "  \n")
        
        y_true = Y_test[y]
        y_pred = predictions[y]
        
        cr = classification_report(y_true, y_pred, output_dict=True)
        
        if cr.get('1'):
            print("Precision: {:.5}, Recall: {:.5},  F1 score : {:.5} ,Accuracy:  {:.5}" 
               .format(cr['1']["precision"], cr["1"]["recall"], cr["1"]["f1-score"], accuracy_score(y_true, y_pred)) )
        else:
            print("Precision: {}, Recall: {},  F1 score : {} ,Accuracy:  {}" 
               .format(1.0, 1.0, 1.0, accuracy_score(y_true, y_pred)) )
        
        print()
        print("#" *20 )


def tokenize(text):
    
    '''
    This function tokenizes text by filtering out only those characters which are upper and lower characters
    
    Words not in stopwords are lemmatized and returned
    '''
    text =re.sub(r'[^a-zA-Z0-9]'," "  , text)
    text = text.lower()
    text = text.split()

    sw= stopwords.words('english')
    
    text   = [word for word in text if not word in sw]
    lemmed_text = [WordNetLemmatizer().lemmatize(word) for word in text]
    return lemmed_text
    
    
def tokenize2(text):
    """ This function tokenizes text by filering out the punctuation characters
    
    Words not in stopwords are lemmatized and returned """
    
    text= [char  if char not in  string.punctuation else " " for char in text.lower()]
    text="".join(text)
    #split text into individual words
    text = text.split()
    # Remove stop words  
    sw= stopwords.words('english')
    text  = [word for word in text if word not in sw]
    lemmed_text = [WordNetLemmatizer().lemmatize(word) for word in text]
    
    return lemmed_text


def train_model(X_train, Y_train):
	"""
	This function accepts a pandas dataframe X_train, 
	with columns "message"  and "genre", 
	and a Y_test which is a multilabeled dataframe
	and performs parameter testing and returns a tuned classifier model


	"""
	categorical_features  = ["genre"]
	categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder())])

	text_features = "message"
	text_transformer =Pipeline(
        steps = [ ("corpus",CountVectorizer(analyzer=tokenize)),
           ("tfid", TfidfTransformer())
        ])

	preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features),
        ('cat', categorical_transformer, categorical_features)])

	end2end = Pipeline(steps = [("preprocessor",preprocessor),
                            ("classifier", MultiOutputClassifier(MultinomialNB()))])
	parameters = {
              "classifier__estimator__alpha"  :[ 0.2, 0.5, 0.8,1.0 ],
              "preprocessor__text__corpus__analyzer": [tokenize, tokenize2],
              "preprocessor__text__corpus__max_features":[1000, 1500],
              "preprocessor__text__tfid__norm": ["l1", "l2", None],
              "preprocessor__text__tfid__use_idf": [True, False],
             }        

	cv_expanded = GridSearchCV(end2end, parameters, verbose=1, cv=6)
	
	return cv_expanded.fit(X_train, Y_train)


if __name__ == '__main__':

	# Define argument list

	parser = argparse.ArgumentParser(description = "allow the user to specify the database entered and the location of the pickle file")

	#Compiling argument list
	parser.add_argument("--database", "-d" ,"--data",  default = 'InsertDatabaseName.db', type = str)
	parser.add_argument("--table", "-t" , default = 'InsertTableName', type = str)
	parser.add_argument("--output", "-o" , "--out", default = "trained_model.pickle", type = str)


	args  = parser.parse_args()

	arg_dict = vars(args)
	


	output_file = arg_dict["output"]
	database_file = 'sqlite:///' + arg_dict["database"]
	table_name = arg_dict["table"]

	#importing database

	engine = create_engine(database_file)
	df = pandas.read_sql(table_name, con=engine)
	Y = df.drop(["message", "original","genre", 'id'], axis = 1)
	X = df[["message", "genre"]]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25)

	model = train_model(X_train, Y_train) 

	make_predictions(model, X_test, Y_test)

	with open(output_file, "wb") as pickle_file:
		pickle.dump(model, pickle_file)

