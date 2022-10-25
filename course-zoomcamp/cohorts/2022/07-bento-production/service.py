import numpy as np

import bentoml
from bentoml.io import JSON, NumpyNdarray

# Pull the model as model reference (it pulls all the associate metadata of the model)
model_ref = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')
# Call DictVectorizer object using model reference
# dv = model_ref.custom_objects['DictVectorizer']
# Create the model runner (it can also scale the model separately)
model_runner = model_ref.to_runner()

# Create the service 'credit_risk_classifier' and pass the model
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])


# Define an endpoint on the BentoML service
@svc.api(NumpyNdarray(), output=NumpyNdarray()) # decorate endpoint as in json format for input and output
def classify(application_data):
    # transform data from client using dictvectorizer
    # vector = dv.transform(application_data)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(application_data)
    
    result = prediction#[0] # extract prediction from 1D array
    print('Prediction:', result)
    return result