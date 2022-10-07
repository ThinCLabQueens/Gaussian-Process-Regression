import os
import requests
import json
import pandas as pd
import pickle as pkl
import numpy as np

vaultsearch = False
if vaultsearch == True:


    url = "http://neurovault.org/api/collections/"

    alldata = []
    while True:
        resp = requests.get(url)

        data = resp.text

        data = json.loads(data)
        
        alldata.append(data['results'])
        if data['next'] != None:
            url = data['next']
        else:
            break
    with open('nvaultsearch.pkl',"wb") as f:
        pkl.dump(alldata,f)
        
        
        
        
with open("nvaultsearch.pkl","rb") as f:
    data = pkl.load(f)
data = [item for sublist in data for item in sublist]
data = pd.DataFrame(data)
data = data[data["number_of_images"] > 0]
data = data.replace(['',None,'nan'], np.nan)
isnul = data.isnull()
data_clean = data[data.columns[~data.isnull().any()]]

print(resp)