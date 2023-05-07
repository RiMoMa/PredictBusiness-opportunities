def computeROC_draw(classifier,cv,X , y,labelOpt):
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.metrics import auc
    from sklearn.metrics import RocCurveDisplay
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.neural_network import MLPClassifier

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    All_y_predict=np.array([])
    All_y_test=np.array([])

    for i, (train, test) in enumerate(cv.split(X, y)):
        from sklearn import preprocessing
        from sklearn.linear_model import Lasso
        scaler = preprocessing.StandardScaler().fit(X[train])
        X_test = X[test]
        y_train = y[train]
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])
       # classifier.fit(X[train], y[train])
        classifier.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_test,
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        y_predict = classifier.predict(X_test)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        All_y_predict = np.concatenate((All_y_predict, y_predict))
        All_y_test = np.concatenate((All_y_test ,y[test]))
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC curve "+labelOpt,
    )
    ax.legend(loc="lower right")
    plt.show()
    return mean_auc , std_auc, All_y_predict, All_y_test

def computeROC_draw_single(classifier,X_train,y_train,X_test,y_test,labelOpt):
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.metrics import auc
    from sklearn.metrics import RocCurveDisplay
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.neural_network import MLPClassifier

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    from sklearn import preprocessing
    from sklearn.linear_model import Lasso
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier.fit(X_train, y_train)
    All_y_predict=classifier.predict(X_test)

    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        name="ROC fold ",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC curve "+labelOpt,
    )
    ax.legend(loc="lower right")
    plt.show()
    return mean_auc , std_auc,All_y_predict,y_test


def computeROC_draw_integration(classifier,cv,X_int,y_int,X ,y,labelOpt):
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.metrics import auc
    from sklearn.metrics import RocCurveDisplay
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.neural_network import MLPClassifier

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    All_y_predict=np.array([])
    All_y_test=np.array([])

    for i, (train, test) in enumerate(cv.split(X, y)):
        from sklearn import preprocessing
        IntegratedData = np.concatenate((X[train],X_int))
        y_integrated = np.concatenate((y[train],y_int))
        scaler = preprocessing.StandardScaler().fit(IntegratedData)
        X_test = X[test]
        y_train = y_integrated
        X_train = scaler.transform(IntegratedData)
        X_test = scaler.transform(X[test])
       # classifier.fit(X[train], y[train])
        classifier.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_test,
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        y_predict = classifier.predict(X_test)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        All_y_predict = np.concatenate((All_y_predict, y_predict))
        All_y_test = np.concatenate((All_y_test ,y[test]))

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC curve "+labelOpt,
    )
    ax.legend(loc="lower right")
    plt.show()
    return mean_auc , std_auc,All_y_predict,All_y_test
