from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from xgboost import XGBClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

#iRec1 = 'fctd.csv'
#iRec2 = 'fconjTRD.csv'
#iRec3 = 'fpsc.csv'

"""ConjointTRD"""
iRecConjoint = 'HybridPSSM_oversample.csv'
D = pd.read_csv(iRecConjoint) #header=None)  # Using pandas
Xtrd = D.iloc[:, :-1].values
ytrd = D.iloc[:, -1].values


pipe = make_pipeline(StandardScaler(), sfs1)
pipe.fit(X_train, y_train)
print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
