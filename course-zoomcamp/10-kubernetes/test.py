import requests

url = 'http://a251ff3a048d54709a4ced34f497b81c-1507187166.us-east-2.elb.amazonaws.com/predict'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)
