import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
import pandas as pd

df = pd.read_csv('Admission_Predict.csv')

toefl_score = df['TOEFL Score'].tolist()

result = df['GRE Score'].tolist()

fig = px.scatter(x = toefl_score, y = result)

fig.show()

import plotly.graph_objects as go

toefl_scores = df['TOEFL Score'].tolist()

chance_of_admission = df['Chance of admit'].tolist()

result = df['GRE Score'].tolist()

colors = []

for data in result:
  if data == 1:
    colors.append('green')
  else:
    colors.append('red')

fig = go.Figure(data = go.Scatter(
    x = toefl_scores,
    y = chance_of_admission,
    mode = 'markers',
    marker = dict(color = colors)
))

fig.show()

factors = df[['TOEFL Score', 'Chance of admit']]

results = df['GRE Score']

toefl_train, toefl_test, results_train, results_test = train_test_split(factors, results, test_size = 0.25, random_state = 0)

print(toefl_train[0:10])

sc_x = StandardScaler()

toefl_train = sc_x.fit_transform(toefl_train)

toefl_test = sc_x.fit_transform(toefl_test)

print(toefl_train[0:10])

results_pred = classifier.predict(toefl_test)

print ("Accuracy : ", accuracy_score(results_test, results_pred)) 

user_score = int(input('Enter the score of the user:'))

user_chance_of_admission = int(input('Enter the chances of admission of the user:'))

user_test = sc_x.transform([[user_score, user_chance_of_admission]])

user_results_pred = classifier.predict(user_test)

if user_purchase_pred[0] == 1:
  print('The user may get admission.')
else:
  print('The user may not get admission.')