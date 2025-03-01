
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score
import sklearn
import pandas as pd
import numpy as np
import joblib
import boto3
import pathlib 
from io import  StringIO
import argparse
import os

# function to load model
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir,"model.joblib"))
    return clf

#sagemaker needs some by default argumnets

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
# hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)


# Data, model, and output directories  parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train_file', type=str, default="train-V-1.csv")
    parser.add_argument('--test_file', type=str, default="test-V-1.csv")
    
    args, _ = parser.parse_known_args()

    print("SKlearn version = ",sklearn.__version__)
    
    print("Joblib version = ",joblib.__version__)

# . .. load from args.train and args.test, train a model, write model to args.model_dir.

    train_df = pd.read_csv(os.path.join(args.train,args.train_file))
    test_df = pd.read_csv(os.path.join(args.test,args.test_file))

    features = list(train_df.columns)

    label = features.pop(-1)

    print("Building training and testing datasets")

    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]
    print()
    print("X_train.shape",X_train.shape)
    print("y_train.shape",y_train.shape)
    print("X_test.shape",X_test.shape)
    print("y_test.shape",y_test.shape)

    print()
    model = RandomForestClassifier(n_estimators = args.n_estimators , random_state = args.random_state, verbose = True)
    model.fit(X_train,y_train)

    print()

    model_path = os.path.join(args.model_dir,"model.joblib")
    joblib.dump(model,model_path)
    print("Model loaded at " + model_path)

    print()

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_classscore = classification_report(y_test,y_pred_test)

    print()

    print("Test Accuracy is",test_acc)
    
    print("Classification Report is",test_classscore)






