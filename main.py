from typing import Literal

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, PositiveInt, PositiveFloat, NonNegativeInt, NonNegativeFloat

import predictor


app = FastAPI()


class ApplicantData(BaseModel):
    Gender: Literal['Male', 'Female']
    Married: Literal['No', 'Yes']
    Dependents: Literal['0', '1', '2', '3+']
    Education: Literal['Not Graduate', 'Graduate']
    Self_Employed: Literal['No', 'Yes']
    ApplicantIncome: PositiveInt | PositiveFloat
    CoapplicantIncome: NonNegativeInt | NonNegativeFloat
    LoanAmount: PositiveInt | PositiveFloat
    Loan_Amount_Term: PositiveInt | PositiveFloat
    Credit_History: Literal['0', '1']

    class Config:
        schema_extra = {
            'example': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '3+',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 5000,
                'CoapplicantIncome': 0,
                'LoanAmount': 1500,
                'Loan_Amount_Term': 120,
                'Credit_History': '1',
            }
        }


class ApplicationDecision(BaseModel):
    bayes: Literal['No', 'Yes']
    tree: Literal['No', 'Yes']
    logistic: Literal['No', 'Yes']
    svm: Literal['No', 'Yes']
    knn: Literal['No', 'Yes']
    voting_majority: Literal['No', 'Yes']

    class Config:
        schema_extra = {
            'example': {
                'bayes': 'No',
                'tree': 'Yes',
                'logistic': 'No',
                'svm': 'Yes',
                'knn': 'Yes',
                'voting_majority': 'Yes',
            }
        }


@app.get('/', response_class=RedirectResponse)
def docs():
    return '/docs'


@app.post('/', response_model=ApplicationDecision)
def predict(x: ApplicantData):
    return predictor.predict(x.dict())
