
import pandas as pd
import yaml
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow.sklearn
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from urllib.parse  import urlparse
import os

from dotenv import load_dotenv
load_dotenv()
print(os.getenv('MLFLOW_TRACKING_URI'))

Tracking_uri=os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_URI']=Tracking_uri

os.environ['MLFLOW_TRACKING_USERNAME']=os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD']=os.getenv('MLFLOW_TRACKING_PASSWORD')

## load the yaml file


def evaluvate(data_path,model_path):
    data=pd.read_csv(data_path)
    x_data=data.drop('loan_status',axis=1)
    y_data=data['loan_status']

    mlflow.set_tracking_uri(Tracking_uri)

    ## load the model from the disk

    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(x_data)
    accuracy=accuracy_score(predictions,y_data)

    mlflow.log_metric('Model accuracy',accuracy)
    print(f'Model Accuracy {accuracy}')

    
params=yaml.safe_load(open('params.yaml'))['train']
if __name__=='__main__':
    evaluvate(params['data_path'],params['model_path'])