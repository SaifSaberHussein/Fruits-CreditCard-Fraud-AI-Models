import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimag
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import mode

catrgories = """0 -> Apple
1 -> Avocado
2 -> Banana
3 -> Blueberry
4 -> Nut"""


train_feat = np.zeros((770, 3))
train_cls = np.zeros(770)
classes = ["Apple", "Avocado", "Banana", "Blueberry", "Nut"]
i = 0
j = 0
for cls in classes:
    path = r"F:\.Coding\Visula Studio Code\Python\Machine Learning\ML Project\Training_project_k-mean"
    path = os.path.join(path, cls)
    for image in os.listdir(path):
        specific_image = os.path.join(path , image)
        
        img = mpimag.imread(specific_image)
        
        img_array = np.array(img)
        
        img_norm = img_array / 255.0
        r_mean = np.mean(img_norm[:,:,0])
        g_mean = np.mean(img_norm[:,:,1])
        b_mean = np.mean(img_norm[:,:,2])
        train_feat[j] = r_mean ,g_mean, b_mean
        train_cls[j] = i
        j += 1
    i += 1
# 33 
kmean = KMeans(n_clusters=5, random_state=33)
kmean.fit(train_feat)
print("Cluster centers:\n", kmean.cluster_centers_)
print("-" * 50)

print("Cluster labels:\n", kmean.labels_)
print("-" * 50)


cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []

for index_train,catch in enumerate(kmean.labels_):

    match catch:
        case 0 :cluster_1.append(train_cls[index_train])
        case 1 :cluster_2.append(train_cls[index_train])
        case 2 :cluster_3.append(train_cls[index_train])
        case 3 :cluster_4.append(train_cls[index_train])
        case 4 :cluster_5.append(train_cls[index_train])

            
def get_class(mode):
    
    if mode[0] == 0:
        return "Apple"
    elif mode[0] == 1:
        return "Avocado"
    elif mode[0] == 2:
        return "Banana"
    elif mode[0] == 3:
        return "Blueberry"
    elif mode[0] == 4:
        return "Nut"
    
cluster_1_mode = mode(cluster_1)
actual_class_1 = get_class(cluster_1_mode)
print("Cluster 1 = " , actual_class_1)

cluster_2_mode = mode(cluster_2)
actual_class_2 = get_class(cluster_2_mode)
print("Cluster 2 = " , actual_class_2)

cluster_3_mode = mode(cluster_3)
actual_class_3 = get_class(cluster_3_mode)
print("Cluster 3 = " , actual_class_3)

cluster_4_mode = mode(cluster_4)
actual_class_4 = get_class(cluster_4_mode)
print("Cluster 4 = " , actual_class_4)

cluster_5_mode = mode(cluster_5)
actual_class_5 = get_class(cluster_5_mode)
print("Cluster 5 = " , actual_class_5)
print("-" * 50)



test_feat = np.zeros((8, 3))
j = 0
path = r"F:\.Coding\Visula Studio Code\Python\Machine Learning\ML Project\Testing_project_k-mean"

for image in sorted(os.listdir(path)):
    specific_image = os.path.join(path , image)
    
    img = mpimag.imread(specific_image)
    
    img_array = np.array(img)
    
    img_norm = img_array / 255.0
    r_mean = np.mean(img_norm[:,:,0])
    g_mean = np.mean(img_norm[:,:,1])
    b_mean = np.mean(img_norm[:,:,2])
    test_feat[j] = r_mean ,g_mean, b_mean

    j += 1
    path = r"F:\.Coding\Visula Studio Code\Python\Machine Learning\ML Project\Testing_project_k-mean"

    
test = kmean.predict(test_feat)

cluster_to_class = {
    0: actual_class_1,
    1: actual_class_2,
    2: actual_class_3,
    3: actual_class_4,
    4: actual_class_5
}


true_classes = ["Blueberry", "Banana" , "Avocado", "Apple", "Nut", "Nut", "Avocado", "Apple"]
correct = 0

for index_test, predict in enumerate(test):
    cluster_id = test[index_test]              
    predicted_class = cluster_to_class[cluster_id]
    print(f"The Test Image number {index_test + 1} is belongs to Cluster {cluster_id + 1} : ", predicted_class)
    if predicted_class == true_classes[index_test]:
        correct += 1
print(f"The Correct Accuracy of Test Image is = {(correct/len(true_classes)) * 100}%")



pred_labels = [cluster_to_class[test[i]] for i in range(len(test))]

cm = confusion_matrix(true_classes, pred_labels, labels=classes)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues", cbar=False)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()




