import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
print(os.getenv('MLFLOW_TRACKING_URI'))

Tracking_uri=os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_URI']=Tracking_uri

os.environ['MLFLOW_TRACKING_USERNAME']=os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD']=os.getenv('MLFLOW_TRACKING_PASSWORD')

def hyper_parameter_tuning(x_train,y_train,param_grid):
    rf=RandomForestClassifier()
    Grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
    Grid_search.fit(x_train,y_train)
    return Grid_search
def Model_training(data_path,model_path,params):
    data=pd.read_csv(data_path)
    print('TRaciking',Tracking_uri)
    mlflow.set_tracking_uri(Tracking_uri)
    with mlflow.start_run():
        x_data=data.drop('loan_status',axis=1)
        y_data=data['loan_status']
        smote = SMOTE(random_state=params['random_state'])
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)
        signature=infer_signature(x_train,y_train)
        # Fit SMOTE to the training data
        X_resampled, y_resampled = smote.fit_resample(x_train, y_train)
        param_grid={
            'n_estimators':params['n_estimators'], 
            'max_depth':params['max_depth'],
        }
        grid_search=hyper_parameter_tuning(X_resampled,y_resampled,param_grid)
        print('mode_train_completed')
        best_model=grid_search.best_estimator_
        y_pred=best_model.predict(x_test)
        accuracy=accuracy_score(y_pred,y_test)
        print(f'accuracy score {accuracy}')
         ## Log additional metrics
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_params(grid_search.best_params_)
        cm=confusion_matrix(y_pred,y_test)
        cr=classification_report(y_pred,y_test)
        mlflow.log_text(str(cm),'Confussion_matrix.txt')
        mlflow.log_text(str(cr),'Classification_report.txt')
        
        tracking_url_type_score=urlparse(mlflow.get_tracking_uri()).scheme  

        if tracking_url_type_score !='file':
            mlflow.sklearn.log_model(best_model,'model',registered_model_name='Best_model')
        else:
            mlflow.sklearn.log_model(best_model,'model',signature=signature)

                ## create the dirctory for the saving the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f'Model has been saved in the path {model_path}')

params=yaml.safe_load(open('params.yaml'))['train']

if __name__=='__main__':
    data_path=params['data_path']
    model_path=params['model_path']
    Model_training(data_path=data_path,model_path=model_path,params=params)
