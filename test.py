import os
os.environ["R_HOME"] = "C:/Users/Ian/anaconda3/envs/gpr/Lib/R"
os.environ["PATH"]   = "C:/Users/Ian/anaconda3/envs/gpr/Lib/R/bin/x64" + ";" + os.environ["PATH"]
from advanced_pca import CustomPCA
import pandas as pd


data=pd.read_csv('output.csv') # Reading datafile (should be in the same directory as our IDE)
PCAdata = data.drop(["Participant #","Runtime_mod","Task_name","Gradient 1","Gradient 2","Gradient 3"],axis=1) # Getting rid of unneeded columns for PCA
PCAmodel = CustomPCA(n_components=4,rotation='varimax') 
PCAmodel.fit(PCAdata)
loadings = PCAmodel.components_.T
names = PCAdata.columns
loadings = pd.DataFrame(loadings,index=names)
PCAresults = PCAmodel.transform(PCAdata).T