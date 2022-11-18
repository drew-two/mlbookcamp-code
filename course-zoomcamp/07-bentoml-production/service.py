import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

from typing import Set
from pydantic import BaseModel
from pydantic import constr

model_ref = bentoml.xgboost.get("credit_risk_model:cacobpdflgrusaav")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

JSONStr = constr(regex='^[A-z]+$')
class CreditApplication(BaseModel):
    seniority: int
    home: JSONStr
    time: int
    age: int
    marital: JSONStr
    records: JSONStr
    job: JSONStr
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int

@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
async def classify(credit_application):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    # return { "status": "Approved"}
    result = prediction[0]

    if result > 0.5:
        return {
            "status": "DECLINED"
        }
    elif result > 0.25:
        return {
            "status": "MAYBE"
        }
    else:
        return {
            "status": "APPROVED"
        }
