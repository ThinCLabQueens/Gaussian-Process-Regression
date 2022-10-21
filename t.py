import lightgbm
from sklearn import datasets,model_selection
data = datasets.make_classification()
#train,test = model_selection.train_test_split(data,train_size=0.8)
model = lightgbm.LGBMClassifier()
model.fit(data[0],data[1])
print('e')
