import pandas as pd
from termcolor import colored
from sklearn.model_selection import train_test_split

# Define variables
COLUMNS = ['created', 'text', 'retwc', 'hashtag', 'followers', 'friends', 'polarity', 'sentiment']

# Read dataset
dataset = pd.read_csv('MlKhattarsent.csv', names = COLUMNS, encoding = 'latin-1')
print(colored("Columns: {}".format(', '.join(COLUMNS)), "yellow"))

# Remove extra columns
print(colored("Useful columns: sentiment and text", "yellow"))
print(colored("Removing other columns", "red"))
dataset.drop(['created', 'retwc', 'hashtag', 'followers', 'polarity', 'friends'], axis = 1, inplace = True)
print(colored("Columns removed", "red"))

# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['sentiment'], test_size = 0.20, random_state = 100)
train_dataset = pd.DataFrame({
	'text': X_train,
	'sentiment': y_train
	})
print(colored("Train data distribution:", "yellow"))
print(train_dataset['sentiment'].value_counts())
test_dataset = pd.DataFrame({
	'text': X_test,
	'sentiment': y_test
	})
print(colored("Test data distribution:", "yellow"))
print(test_dataset['sentiment'].value_counts())
print(colored("Split complete", "yellow"))

# Save train data
print(colored("Saving train data", "yellow"))

train_dataset.to_csv('MlKhattarsenttrain.csv', index = False)
print(colored("Train data saved to data/train.csv", "green"))

# Save test data
print(colored("Saving test data", "yellow"))
test_dataset.to_csv('MlKhattarsenttest.csv', index = False)
print(colored("Test data saved to data/test.csv", "green"))