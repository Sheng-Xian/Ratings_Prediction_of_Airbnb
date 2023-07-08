import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import precision_recall_fscore_support, plot_confusion_matrix, roc_curve, auc, \
    multilabel_confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df_raw = pd.read_csv("data/listings.csv")
df = df_raw.copy()


df_cleaned = pd.read_csv("data/listings_cleaned_data.csv")

#
numerical_columns = ['host_listings_count', 'bathrooms_text', 'bedrooms', 'beds','accommodates', 'price',
                     'number_of_reviews','reviews_per_month', 'minimum_nights', 'maximum_nights','first_review_since',
                     'last_review_since','host_days_active', 'latitude','longitude']
scaler = StandardScaler()
# temp1 = df_cleaned.drop(columns=numerical_columns)
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
# X = subset_neighborhood.drop(columns="review_scores_location")
# y = subset_neighborhood["review_scores_location"]

def formatting(value):
    if value < 4.6:
        return 0
    elif value < 4.82:
        return 1
    elif value < 5:
        # if value < 5:
        return 2
    else:
        return 3

df_cleaned['review_scores_rating'] = df_cleaned['review_scores_rating'].apply(formatting)
df_cleaned['review_scores_rating'].hist()
plt.show()
df_cleaned.drop('id', axis=1, inplace=True)

df_y = df_cleaned["review_scores_rating"]
df_X = df_cleaned.drop(columns=["review_scores_rating","review_scores_accuracy", "review_scores_cleanliness",
                                "review_scores_checkin","review_scores_communication","review_scores_location", "review_scores_value"], axis=1)
X = df_X[df_X.columns].to_numpy()
y = df_cleaned["review_scores_rating"].to_numpy()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=50)

def dummy() :
    classifier = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    classifier.fit(Xtrain, ytrain)
    print("----Baseline Classifier: Most frequent----")
    print(classifier.score(Xtrain, ytrain))
    print(classifier.score(Xtest, ytest))


def knn():
    k_range = [1]
    # k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 30]
    mean_error = []
    std_error = []
    for k in k_range:
        classifier = KNeighborsClassifier(n_neighbors=k,metric="minkowski") #weights='uniform'
        classifier.fit(Xtrain, ytrain)
        ypred = classifier.predict(Xtest)
        print("-----knn-----")
        print("Classification report: " + classification_report(ytest, ypred))
        print("Accuracy: " + f"{accuracy_score(ytest, ypred)}")
        # from sklearn.model_selection import cross_val_score
    #     scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    #     mean_error.append(np.array(scores).mean())
    #     std_error.append(np.array(scores).std())
    # plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    # plt.xlabel('k'); plt.ylabel('F1 Score')
    # plt.title("KNeighborsClassifier: Different F1 score for different k")
    # plt.show()


def logistic_regression() :
    # lr = LogisticRegression(penalty='none',solver='newton-cg', max_iter=1000)
    lr = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=0.1)
    lr.fit(Xtrain, ytrain)
    print("----logistic regression----")
    print(lr.score(Xtrain, ytrain))
    print(lr.score(Xtest, ytest))
    ypred = lr.predict(Xtest)
    print(classification_report(ytest, ypred))
    #Plot the features importance ranking by coef, if coef is bigger then means that the feature is more important.
    coef_c1 = pd.DataFrame({'var': pd.Series(df_X.columns),
                            'coef_abs': abs(pd.Series(lr.coef_[0].flatten()))
                            })
    coef_c1 = coef_c1.sort_values(by='coef_abs', ascending=False)
    print(coef_c1)


def plot_knn_k_selection():
    precRecallF1Class1 = [];
    precRecallF1Class2 = [];
    precRecallF1Class3 = [];
    precRecallF1Class4 = []
    stdErr1 = [];
    stdErr2 = [];
    stdErr3 = [];
    stdErr4 = []

    k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    for k in k_range:
        print("> Tree Depth %.1f" % k)
        kf = KFold(n_splits=5)
        temp1 = [];
        temp2 = [];
        temp3 = [];
        temp4 = []
        meanAccuracy = []
        for train, test in kf.split(X):
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X[train], y[train])
            # model = TreeClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=t, random_state=42)
            # model.fit(X[train], ytrain[train])
            ypred = model.predict(X[test])
            pred = np.array(precision_recall_fscore_support(y[test], ypred))
            temp1.append(pred[0]);
            temp2.append(pred[1]);
            temp3.append(pred[2]);
            temp4.append(pred[3])
            meanAccuracy.append(accuracy_score(y[test], ypred))

        print("\tAccuracy = %f\n" % np.array(meanAccuracy).mean())
        precRecallF1Class1.append(np.array(temp1).mean(axis=0))
        precRecallF1Class2.append(np.array(temp2).mean(axis=0))
        precRecallF1Class3.append(np.array(temp3).mean(axis=0))
        precRecallF1Class4.append(np.array(temp4).mean(axis=0))
        stdErr1.append(np.array(temp1).std(axis=0))
        stdErr2.append(np.array(temp2).std(axis=0))
        stdErr3.append(np.array(temp3).std(axis=0))
        stdErr4.append(np.array(temp4).std(axis=0))

    precRecallF1Class1 = np.array(precRecallF1Class1)
    precRecallF1Class2 = np.array(precRecallF1Class2)
    precRecallF1Class3 = np.array(precRecallF1Class3)
    precRecallF1Class4 = np.array(precRecallF1Class4)
    stdErr1 = np.array(stdErr1)
    stdErr2 = np.array(stdErr2)
    stdErr3 = np.array(stdErr3)
    stdErr4 = np.array(stdErr4)

    plt.errorbar(k_range, precRecallF1Class1[:, 3], yerr=stdErr1[:, 3], label='Class 1')
    plt.errorbar(k_range, precRecallF1Class2[:, 3], yerr=stdErr2[:, 3], label='Class 2')
    plt.errorbar(k_range, precRecallF1Class3[:, 3], yerr=stdErr3[:, 3], label='Class 3')
    plt.errorbar(k_range, precRecallF1Class4[:, 3], yerr=stdErr4[:, 3], label='Class 4')
    plt.title("KNeighborsClassifier: Different F1 score for different k")
    plt.xlabel('Quantity of K')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


#Failed to implement
def plot_logistic_regression_C_selection():
    Ci_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    f1_class1 = [];  f1_class2 = [];  f1_class3 = []; f1_class4 = []
    std_error0 = []; std_error1 = []; std_error2 = []; std_error3 = []

    for Ci in Ci_range:
        k = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []; temp4 = []
        meanAccuracy = []
        for train, test in k.split(X):
            model = LogisticRegression(multi_class="multinomial", penalty='l2', solver='lbfgs', max_iter=100000, C=Ci, random_state=50)
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            pred = np.array(precision_recall_fscore_support(y[test], ypred))
            temp1.append(pred[0]), temp2.append(pred[1]), temp3.append(pred[2]), temp4.append(pred[3])
            meanAccuracy.append(accuracy_score(y[test], ypred))

        f1_class1.append(np.array(temp1).mean(axis=0))
        f1_class2.append(np.array(temp2).mean(axis=0))
        f1_class3.append(np.array(temp3).mean(axis=0))
        f1_class4.append(np.array(temp4).mean(axis=0))
        std_error0.append(np.array(temp1).std(axis=0))
        std_error1.append(np.array(temp2).std(axis=0))
        std_error2.append(np.array(temp3).std(axis=0))
        std_error3.append(np.array(temp4).std(axis=0))

    f1_class1 = np.array(f1_class1)
    f1_class2 = np.array(f1_class2)
    f1_class3 = np.array(f1_class3)
    f1_class4 = np.array(f1_class4)
    std_error0 = np.array(std_error0)
    std_error1 = np.array(std_error1)
    std_error2 = np.array(std_error2)
    std_error3 = np.array(std_error3)
    print(f1_class1)
    print(std_error0)
    plt.errorbar(Ci, f1_class1, yerr=std_error0, label='Class 1')
    plt.errorbar(Ci, f1_class2, yerr=std_error1, label='Class 2')
    plt.errorbar(Ci, f1_class3, yerr=std_error2, label='Class 3')
    plt.errorbar(Ci, f1_class4, yerr=std_error3, label='Class 4')
    plt.title("C cross-validation")
    plt.xlabel('C')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


def plot_roc() :
    n_classes = 4
    lw = 2
    # Draw ROC for baseline classifier
    baseline_classifier = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    yproba_bs = baseline_classifier.fit(Xtrain, ytrain).predict_proba(Xtest)
    ytestROC_bs = label_binarize(ytest, classes=[0, 1, 2, 3])
    fpr_bs = dict()
    tpr_bs = dict()
    roc_auc_bs = dict()
    for i in range(n_classes):
        fpr_bs[i], tpr_bs[i], _  = roc_curve(ytestROC_bs[:, i], yproba_bs[:, i])
        roc_auc_bs[i] = auc(fpr_bs[i], tpr_bs[i])
    fpr_bs["micro"], tpr_bs["micro"], _ = roc_curve(ytestROC_bs[:, i], yproba_bs[:, i])
    roc_auc_bs["micro"] = auc(fpr_bs["micro"], tpr_bs["micro"])
    plt.plot(fpr_bs[2], tpr_bs[2], color='navy',
             lw=lw, label='Baseline ROC curve (area = %0.2f)' % roc_auc_bs[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    model = KNeighborsClassifier(n_neighbors=1, weights='uniform').fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

    # Compute ROC curve and ROC area for all classes of logistic regression
    lr_lassifier = OneVsRestClassifier(LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=1))
    ypred_lr = model.predict(Xtest)
    yproba_lr = lr_lassifier.fit(Xtrain, ytrain).predict_proba(Xtest)
    ytestROC_lr = label_binarize(ytest, classes=[0,1,2,3])
    fpr_lr = dict()
    tpr_lr = dict()
    roc_auc_lr = dict()
    for i in range(n_classes):
        fpr_lr[i], tpr_lr[i], _  = roc_curve(ytestROC_lr[:, i], yproba_lr[:, i])
        roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])
    # Compute micro-average ROC curve and ROC area for logistic regression
    fpr_lr["micro"], tpr_lr["micro"], _ = roc_curve(ytestROC_lr[:, i], yproba_lr[:, i])
    roc_auc_lr["micro"] = auc(fpr_lr["micro"], tpr_lr["micro"])
    plt.plot(fpr_lr[2], tpr_lr[2], color='darkorange',
             lw=lw, label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc_lr[2])

    # Compute ROC curve and ROC area for all classes of KNN
    knnClassifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1, weights='uniform'))
    yproba = knnClassifier.fit(Xtrain, ytrain).predict_proba(Xtest)
    ytestROC = label_binarize(ytest, classes=[0, 1, 2, 4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytestROC[:, i], yproba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area for kNN
    fpr["micro"], tpr["micro"], _ = roc_curve(ytestROC[:, i], yproba[:, i])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr[2], tpr[2], color='green',
             lw=lw, label='KNN ROC curve (area = %0.2f)' % roc_auc[2])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (C=' + str(1) + ', Poly=' + str(1) + ')')
    plt.legend(loc="lower right")
    plt.show()

    # model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=1).fit(Xtrain,ytrain)

    # plot confusion matrix
    print("\nBaseline classifier\n")
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    ydummy = dummy.predict(Xtest)
    print("Model classes:", model.classes_)
    print("Baseline confusion matrix:")
    print(multilabel_confusion_matrix(ytest, ydummy))
    print( accuracy_score(ytest, ydummy) )

    print("Logistic Regression Confusion Matrix:")
    print(multilabel_confusion_matrix(ytest, ypred_lr))
    print( accuracy_score(ytest, ypred_lr))

    print("kNN confusion matrix:")
    print(multilabel_confusion_matrix(ytest, ypred))
    print(accuracy_score(ytest, ypred))



dummy()
logistic_regression()
knn()
plot_roc()
plot_knn_k_selection()
