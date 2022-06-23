import pickle
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc
#
# Create your model here (same as above)
accuray = []
auROC = []
avePrecision = []
F1_Score = []
AUC = []
MCC = []
Recall = []
CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)



# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model= pickle.load(file)

# Calculate the accuracy score and predict target values
#score = pickle_model.score(Xtest, Ytest)
#print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(Xtest)
accuray.append(accuracy_score(y_pred=Ypredict , y_true=Ytest))
print ("Accuracy" %f, accuracy) 
