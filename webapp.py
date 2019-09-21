from flask import Flask,  render_template,  request, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import StringField , SubmitField,  RadioField, TextField
from wtforms.validators import DataRequired
import pandas, pickle, matplotlib,  seaborn, base64, json, plotly
import plotly.graph_objs as go 
#from io import StringIO
from matplotlib import pyplot
from sqlalchemy import create_engine
from train import tokenize, tokenize2

app = Flask(__name__)

app.config["SECRET_KEY"] = 'mysecretkey'


class InfoForm(FlaskForm):
	#grabs information about puppies
	input_message = StringField("Input the message" , validators = [DataRequired()])
	
	submit  = SubmitField("Submit")
							
	genre = RadioField("Genre",
		 choices = [("direct", "direct"),
		 			("news", "news"),
		 			("social", "social")], default= "direct")
		 			

def category_plot(df):
	"""

	This function accepts the dataframe and creates a plot that shows the count of the 
	categories respresented in the messages arranged by their source.

	"""
	
	group_df = df.groupby("genre").sum()

	traces = [go.Bar(
    x = group_df.columns,
    y = group_df.loc[name], name = name) for name in group_df.index]

	return json.dumps(traces, cls=plotly.utils.PlotlyJSONEncoder)


def related_plot(df):

	""" This function accepts the data farm and creates a plot that 
	showes the number of messages that are related or not related 
	arranged  by the  source of the message """

	df_zero = df[df["related"] == 0] [["genre", "related"]].groupby("genre").count()
	df_zero.columns = ["Nonrelevant"]
	df_one = df[df["related"] == 1] [["genre", "related"]].groupby("genre").count()
	df_one.columns = ["Relavant"] 

	new_df = pandas.concat([df_zero, df_one], axis  = 1)

	traces = [go.Bar(
    x = new_df.columns,
    y = new_df.loc[name],
    name = name) for name in new_df.index]


	return json.dumps(traces, cls=plotly.utils.PlotlyJSONEncoder)	


def load_df(db='sqlite:///InsertDatabaseName.db', table = 'InsertTableName' ):
	"""
	reads a database from the an SQL file"""
	engine = create_engine(db)
	df = pandas.read_sql(table, con=engine)
	#Remove unnecessary columns
	return df.drop(["id", "original"], axis = 1)


def load_model(model_file  = "trained_model.pickle"):
	"""" loads a trained model from a pickle file """
	with open(model_file, "rb") as pickle_file:
		model =  pickle.load(pickle_file)
	return model
							
	
@app.route('/', methods  = ["GET", "POST"])
def index():

	print("session is :", type(session))

	input_message = False
	genre = "news"

	form = InfoForm()

	if form.validate_on_submit():
		session["genre"] = form.genre.data
		session["input_message"] = form.input_message.data
				
		return redirect(url_for("evaluate"))#

	rel = related_plot(df)
	c_p = category_plot(df.drop("related", axis = 1))	
	
	return render_template("index.html",  category_plot = c_p, related_plot = rel, form =form,)


@app.route("/evaluate", methods  = ["GET", "POST"])
def evaluate():
	""" """

	input_message = False
	genre = "news"

	form = InfoForm()

	if form.validate_on_submit():
		session["genre"] = form.genre.data
		session["input_message"] = form.input_message.data
				
		return redirect(url_for("evaluate"))#

	model = load_model()

	prediction = model.predict(pandas.DataFrame({"genre":[ session["genre"]], "message": [session["input_message"]]}))
	prediction = pandas.DataFrame(prediction, columns = df_labels)

	results = prediction.iloc[0][prediction.iloc[0] == 1].index.tolist()

	if (len(results) == 0) or (results == ['related']):
		results = ["This message is not relevant"]

	return render_template("process.html", results=results, form = form)


if __name__ == '__main__':

	df = load_df()
	df_labels = df.drop(["message", "genre"]  ,axis =1).columns.tolist() 
	app.run()