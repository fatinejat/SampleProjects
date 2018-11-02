import numpy as np
import OwnITK as oitk
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto')
X = []

training_set = []

imgData, tmpDIM, tmpSpaces = oitk.getItkImageData('HGG_brats_2013_pat0004_1/' + 'Flair.nii')
img = imgData.flatten()

for i in range(10):
    training_set.append(imgData)
    X.append(np.random.randint(2, size=img.shape))

pca = PCA(copy=True)
X = pca.fit_transform(X)
test_data = pca.transform(img)


result = np.zeros(tmpDIM)
print result.shape
print imgData.shape

for i in range(tmpDIM[0]):
    for j in range(tmpDIM[1]):
        for k in range(tmpDIM[2]):
            y = np.zeros(len(training_set))
            for num in range(len(training_set)):
                # print str(i) + ',' + str(j) + ',' + str(k) + ': ' + str(training_set[num][k][j][i])
                y[num] = training_set[num][k][j][i]

            random_forest.fit(X, y)
            result[k][j][i] = random_forest.predict(test_data)
            #print str(i) + ',' + str(j) + ',' + str(k) + ': ' + str(y[0]) + ' -> ' + str(result[k][j][i])
