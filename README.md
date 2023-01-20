# disaster-response


Installations

The code should run with no issues using Python versions 3.*. 
Libraries used:
pandas
numpy
nltk
flask
plotly
json
sklearn
sys 
re
pickle
sqlalchemy


About project

We used message about disaster to build ML model that classifies messages to specific categories in real time. Web app shows the result of the codes.
The screenshots of web app provided below.

File Descriptions

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py

- models
|- train_classifier.py
|- classify.pkl  # saved model 

- README.md

Instructions

Run the following commands in the project's root directory to set up database and model:

To run ETL pipeline that cleans data and stores in database 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disasterresponse.db'
To run ML pipeline that trains classifier and saves 'python models/train_classifier.py data/disasterresponse.db models/classify.pkl'

Run the following command in the app's directory to run web app 'python run.py'

Go to http://0.0.0.0:3001/


Must give credit to Figure 8 for the data and Udacity
This project is part of Udacity nanodegree program

SCREENSHOTS

You need to write disaster message and click 'classify message':

![image](https://user-images.githubusercontent.com/120465527/213779570-61d8de4f-00ba-4a6f-8eaf-f39ec9a596a7.png)

and see results:

![image](https://user-images.githubusercontent.com/120465527/213779932-9f41131b-1075-4ca9-b49e-cd5767d2473b.png)

Additionally, you can see some graphs describing data:

<img width="1194" alt="image" src="https://user-images.githubusercontent.com/120465527/213780331-aec99c00-bcd2-4462-a9b9-49ccbabf5072.png">
