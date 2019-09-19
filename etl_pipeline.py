import argparse, pandas
from sqlalchemy import create_engine

def convert_message(dataset, data_column):
	''' 
	This function accept a dataset and a column name and returns a clean machine learnable dataset
	'''

	categories = dataset[data_column].str.split(";", expand=True)
	#Split this column 

	# use the first fow to extract a list of column names for the dataset
	row = categories.iloc[0]
	category_colnames = list(map(lambda c : c[:-2], list(row)))

	categories.columns = category_colnames

	for column in categories:
		# set each value to be the last character of the string
		# modulo 2 would convert 2s to 0s and preserve 1s and 0
		categories[column] = categories[column].apply(lambda c:int(c[-1])%2)

    #drop the original columns and concatenate expanded columns

	dataset.drop(data_column, inplace=True, axis=1)
	dataset = pandas.concat([dataset, categories], axis=1)
	dataset.drop_duplicates(inplace=True)
	return dataset










if __name__ == '__main__':

	# Define argument list
	parser = argparse.ArgumentParser(description = "allow the user to specify the loction of the input and poutpu")

	#Compiling argument list
	parser.add_argument("--message", "-m" ,"--messages",  default = './data/disaster_messages.csv', type = str)
	parser.add_argument("--categories", "-c" , "--cat", default = "./data/disaster_categories.csv", type = str)
	parser.add_argument("--database", "-d", "--data", default= "InsertDatabaseName.db",  type = str)
	parser.add_argument("--table", "-t", default = 'InsertTableName', type = str)
	
	#Parse arguments
	args  = parser.parse_args()

	arg_dict = vars(args)
	
	messages_file = arg_dict["message"]
	category_file = arg_dict["categories"]
	database_file = 'sqlite:///' + arg_dict["database"]
	table_name = arg_dict["table"]

	#load the messages dataset 
	messages = pandas.read_csv('./data/disaster_messages.csv')

	# load categories dataset
	categories = pandas.read_csv("./data/disaster_categories.csv")
	# merge the datasets
	df = pandas.merge(categories, messages, on="id")

	#process the categories column
	df = convert_message(df, "categories")
	

	#Store database.
	engine = create_engine(database_file)
	df.to_sql(table_name, engine, index=False)

	

