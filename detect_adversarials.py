# print('Load modules...')
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score, precision_score
# import argparse

# # Processing the arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
# parser.add_argument("--detector", default='InputMFS', help="the detector you want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
# parser.add_argument("--net", default='mnist', help="the network used for the attack, either mnist, cif10, or cif100")
# parser.add_argument("--mode", default='test', help="choose test or validation case")
# args = parser.parse_args()

# # Choose attack
# attack_method = args.attack
# detector = args.detector
# mode = args.mode
# net = args.net

# # Load characteristics
# print('Loading characteristics...')
# characteristics = np.load(f'./data/characteristics/{net}_{attack_method}_{detector}.npy', allow_pickle=True)
# characteristics_adv = np.load(f'./data/characteristics/{net}_{attack_method}_{detector}_adv.npy', allow_pickle=True)

# shape = np.shape(characteristics)
# k = shape[0]

# # Split data into training and testing/validation sets
# adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=0.2, random_state=42)
# b_X_train_val, b_X_test, b_y_train_val, b_y_test = train_test_split(characteristics, np.zeros(k), test_size=0.2, random_state=42)
# adv_X_train, adv_X_val, adv_y_train, adv_y_val = train_test_split(adv_X_train_val, adv_y_train_val, test_size=0.2, random_state=42)
# b_X_train, b_X_val, b_y_train, b_y_val = train_test_split(b_X_train_val, b_y_train_val, test_size=0.2, random_state=42)

# X_train = np.concatenate((b_X_train, adv_X_train))
# y_train = np.concatenate((b_y_train, adv_y_train))

# if mode == 'test':
#     X_test = np.concatenate((b_X_test, adv_X_test))
#     y_test = np.concatenate((b_y_test, adv_y_test))
# elif mode == 'validation':
#     X_test = np.concatenate((b_X_val, adv_X_val))
#     y_test = np.concatenate((b_y_val, adv_y_val))
# else:
#     print('Not a valid mode')
#     exit()

# # Function to evaluate and print results for a classifier
# def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name):
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     prediction_pr = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else prediction

#     acc = accuracy_score(y_test, prediction)
#     precision = precision_score(y_test, prediction)
#     recall = recall_score(y_test, prediction)
#     f1 = f1_score(y_test, prediction)
#     auc = roc_auc_score(y_test, prediction_pr)

#     print(f'\nResults for {name}:')
#     print(f'Accuracy: {acc * 100:.2f}%')
#     print(f'Precision: {precision * 100:.2f}%')
#     print(f'Recall: {recall * 100:.2f}%')
#     print(f'F1 Score: {f1 * 100:.2f}%')
#     print(f'AUC Score: {auc * 100:.2f}%')

# # Train and evaluate classifiers
# print('Training and evaluating classifiers...')

# # Logistic Regression
# lr = LogisticRegression()
# evaluate_classifier(lr, X_train, y_train, X_test, y_test, "Logistic Regression")

# # K-Nearest Neighbors
# knn = KNeighborsClassifier()
# evaluate_classifier(knn, X_train, y_train, X_test, y_test, "K-Nearest Neighbors")

# # Gaussian Naive Bayes
# gnb = GaussianNB()
# evaluate_classifier(gnb, X_train, y_train, X_test, y_test, "Gaussian Naive Bayes")

# # Decision Tree
# dt = DecisionTreeClassifier()
# evaluate_classifier(dt, X_train, y_train, X_test, y_test, "Decision Tree")

# # Random Forest
# rf = RandomForestClassifier()
# evaluate_classifier(rf, X_train, y_train, X_test, y_test, "Random Forest")

# # SVM
# scaler = MinMaxScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# svm = SVC(probability=True)
# evaluate_classifier(svm, X_train_scaled, y_train, X_test_scaled, y_test, "SVM")

# # Save the best performing classifier
# best_clf = lr  # Assuming Logistic Regression is the best, replace based on actual results
# filename = f'./data/detectors/Best_Classifier_{attack_method}_{detector}_{mode}_{net}.sav'
# pickle.dump(best_clf, open(filename, 'wb'))
# print(f'Best performing classifier saved as {filename}')

####cif10and 100
print('Load modules...')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse

# Processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='InputMFS', help="the detector you want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
parser.add_argument("--mode", default='test', help="choose test or validation case")
args = parser.parse_args()

# Choose attack
attack_method = args.attack
detector = args.detector
mode = args.mode
net = args.net
scale = True

# Load characteristics
print('Loading characteristics...')
characteristics = np.load(f'./data/characteristics/{net}_{attack_method}_{detector}.npy', allow_pickle=True)
characteristics_adv = np.load(f'./data/characteristics/{net}_{attack_method}_{detector}_adv.npy', allow_pickle=True)

shape = np.shape(characteristics)
k = shape[0]

adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=0.2, random_state=42)
b_X_train_val, b_X_test, b_y_train_val, b_y_test = train_test_split(characteristics, np.zeros(k), test_size=0.2, random_state=42)
adv_X_train, adv_X_val, adv_y_train, adv_y_val = train_test_split(adv_X_train_val, adv_y_train_val, test_size=0.2, random_state=42)
b_X_train, b_X_val, b_y_train, b_y_val = train_test_split(b_X_train_val, b_y_train_val, test_size=0.2, random_state=42)

X_train = np.concatenate((b_X_train, adv_X_train))
y_train = np.concatenate((b_y_train, adv_y_train))

if mode == 'test':
    X_test = np.concatenate((b_X_test, adv_X_test))
    y_test = np.concatenate((b_y_test, adv_y_test))
elif mode == 'validation':
    X_test = np.concatenate((b_X_val, adv_X_val))
    y_test = np.concatenate((b_y_val, adv_y_val))
else:
    print('Not a valid mode')
    exit()

# Function to evaluate and print results for a classifier
def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name):
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    prediction_pr = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else prediction

    acc = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    auc = roc_auc_score(y_test, prediction_pr)

    print(f'\nResults for {name}:')
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')
    print(f'AUC Score: {auc * 100:.2f}%')

# Train and evaluate classifiers
print('Training and evaluating classifiers...')

# Logistic Regression
lr = LogisticRegression()
evaluate_classifier(lr, X_train, y_train, X_test, y_test, "Logistic Regression")

# K-Nearest Neighbors
knn = KNeighborsClassifier()
evaluate_classifier(knn, X_train, y_train, X_test, y_test, "K-Nearest Neighbors")

# Gaussian Naive Bayes
gnb = GaussianNB()
evaluate_classifier(gnb, X_train, y_train, X_test, y_test, "Gaussian Naive Bayes")

# Decision Tree
dt = DecisionTreeClassifier()
evaluate_classifier(dt, X_train, y_train, X_test, y_test, "Decision Tree")

# Random Forest
rf = RandomForestClassifier()
evaluate_classifier(rf, X_train, y_train, X_test, y_test, "Random Forest")

# SVM
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = SVC(probability=True)
evaluate_classifier(svm, X_train_scaled, y_train, X_test_scaled, y_test, "SVM")

# Save the best performing classifier (example assumes Logistic Regression)
best_clf = lr  # Replace with actual best-performing classifier based on results
filename = f'./data/detectors/Best_Classifier_{attack_method}_{detector}_{mode}_{net}.sav'
pickle.dump(best_clf, open(filename, 'wb'))
print(f'Best performing classifier saved as {filename}')
