from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Skeletal open-source code.

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = KNeighborsClassifier()

train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

test_x = images[10000:10100]
expected = labels[10000:10100].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))