from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euclidean_dist(a,b):
    return distance.euclidean(a,b)

class KNN_Scratch():

    def fit(self,training_data,target_data):
        self.training_data=training_data
        self.target_data=target_data

    def predict(self,test_data):
        predictions=[]
        for row in test_data:
            label=self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_distance=euclidean_dist(row,self.training_data[0])
        best_index=0
        for i in range(1,len(self.training_data)):
            dist=euclidean_dist(row,self.training_data[i])
            if dist<best_distance:
                best_distance=dist
                best_index=i
        return self.target_data[best_index]

def KNeighbor():

    border="-"*50
    iris=load_iris()

    data=iris.data
    target=iris.target
    print(border)
    print('Actual data set: ')
    print(border)
    for i in range(len(iris.target)):
        print("ID: %d, Label: %s, Feature: %s"%(i,iris.data[i],iris.target[i]))
    print("Size of Actual dataset: %d"%(i+1))

    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
    print(border)
    print("Training Data set")
    print(border)

    for i in range(len(data_train)):
        print("ID: %d, Label: %s, Feature: %s" % (i, data_train[i], target_train[i]))
    print("Size of Training dataset: %d" % (i + 1))

    print(border)
    print("Target data set")
    print(border)
    for i in range(len(data_train)):
        print("ID: %d, Label: %s, Feature: %s" % (i, data_test[i], target_test[i]))
    print("Size of Testing dataset: %d" % (i + 1))

    classifier=KNN_Scratch()
    classifier.fit(data_train,target_train)
    predictions=classifier.predict(data_test)
    accuracy=accuracy_score(target_test,predictions)
    return accuracy

def main():
    accuracy=KNeighbor()
    print("Accuracy of classification algorithm with K Neighbor classifier is ",accuracy*100,"%")

if __name__=="__main__":
    main()

