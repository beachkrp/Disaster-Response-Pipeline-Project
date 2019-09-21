# Disaster Response Pipeline Project
### Installation

git clone https://github.com/beachkrp/Disaster-Response-Pipeline-Project.git


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        python etl_pipeline.py [-m --messages  [data/disaster_messages.csv] ] [-c --cat [data/disaster_categories.csv]] [-d -- data	[InsertDatabaseName.db]]  [-t --table [InsertTableName]]
    
	
	- To run ML pipeline that trains classifier and saves
        `
		python train.py [--database -d --data [InsertDatabaseName.db]] [--table -t [InsertTableName]] [--output -o  --out [trained_model.pickle]]

2. Run the following command in the app's directory to run your web app.
    python webapp.py

3. Go to http://127.0.0.1:5000/

### Using Libraries
Python 3.4.3
NLTK 3.4
Sci-kit Learn 0.20.4
Flask 0.12.2
Plotly 3.4.2
Dash 0.21.1.

### Acknowledgements. 

https://stackoverflow.com/questions/54541490/sklearn-text-and-numeric-features-with-columntransformer-has-value-error/57970935#57970935

Python Data Science and Machine Learning Bootcamp,Pierian Data
Python and Flask Bootcamp,  Pierian Data.
The Python Mega Course, Ardit Sulce
Interactive Dashboards With Plotly and Dash, Pierian Data

https://blog.heptanalytics.com/2018/08/07/flask-plotly-dashboard/