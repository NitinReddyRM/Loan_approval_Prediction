import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

params=yaml.safe_load(open('params.yaml'))['preprocess']

def Categorical_data(data):
    fig, ax = plt.subplots(1, 4, figsize=(12,3))
    ax = ax.flatten()
    column = 0
    for categorical in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
        # Only select top 8 values, if there are more than 8 unique values
        bar_plot = pd.DataFrame(data[categorical].value_counts().sort_values(ascending=False)).reset_index().head(8)
        bar_plot.columns = ['category', 'count']
        bar_plot['percentage'] = bar_plot['count'] / data.shape[0] * 100

        ax[column].bar(x=bar_plot['category'], height=bar_plot['percentage'], fill=False, edgecolor='tab:blue')
        ax[column].set_title(categorical + ' (%)')
        ax[column].set_xticklabels(bar_plot['category'], rotation=45, ha='right')

        column += 1

    plt.tight_layout()
    plt.savefig('Image/Categorical_Graph.png',dpi=300)

def Numerical_graph(data):
    num_cols = 3
    num_rows = (len([col for col in data.columns if data[col].dtype != 'object']) + num_cols - 1) // num_cols
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    ax = ax.flatten()

    # Plot boxplots
    index = 0
    for col in data.columns:
        if data[col].dtype != 'object':
            ax[index].boxplot(data[col])
            ax[index].set_title(col)
            index += 1

    # Remove any unused axes
    for i in range(index, len(ax)):
        fig.delaxes(ax[i])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Image/boxplots.png')

    #Save Graph for categorical data

def preprocessing(input_path,output_path,image):
    data=pd.read_csv(input_path)
    os.makedirs(image,exist_ok=True)
    Categorical_data(data=data)
    Numerical_graph(data=data)
    encoder=LabelEncoder()
    for i in data.columns:
        if data[i].dtypes=='object':
            data[i]=encoder.fit_transform(data[i])
    data['income_to_loan_ratio'] = data['person_income'] / data['loan_amnt']

#Loan Amount per Year of Employment
    data['loan_per_emp_year'] = data['loan_amnt'] / (data['person_emp_length'] + 1)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path,index=False)

    print(f'preocessing data saved to {output_path}')
if __name__=='__main__':
    preprocessing(params['input'],params['output'],params['images'])

