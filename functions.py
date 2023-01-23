# Functions for Travel Insurance Prediction Project
# Here I created several plotting functions for my Machine Learning project.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


def plot_col_distribution(df, col, bins):
    """Function that plots a histogram for a particular feature in the dataset"""
    _, axs = plt.subplots(1, 1, figsize=(14, 8), tight_layout=True)

    df = df.sort_values(by=col)
    N, _, patches = axs.hist(df[col], bins=bins)

    fracs = (N ** (1 / 5)) / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.PuBu(norm(thisfrac + 0.00015))
        thispatch.set_facecolor(color)

    plt.xlabel(col)
    plt.ylabel("Number of occurences")
    plt.title(f"{col} Distribution", fontsize=15)
    plt.show()


def plot_var_with_travel_insurance(df, col, description):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.barplot(
        x=df.TravelInsurance.value_counts().index,
        y=df.TravelInsurance.value_counts(),
        palette=["#ab4652", "#4565ad"],
        ax=ax[0],
    )
    ax[0].set_title(f"Number Of entries By {col}")
    ax[0].set_ylabel("Count")
    ax[0].grid(False)

    sns.countplot(
        data=df,
        x=col,
        hue="TravelInsurance",
        palette=[
            "#ab4652",
            "#4565ad"])
    ax[1].set_title(
        f"{col}: Bought Travel Insurance vs Did not buy Travel Insurance")
    ax[1].grid(False)

    f.suptitle(f"{description} and whether they bought insurance or not")
    plt.show()


def plot_violin_plot_with_hue(data, x, y, hue):
    plt.subplots(1, 1, figsize=(13, 7), tight_layout=True)
    sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        split=True,
        palette=[
            "#ab4652",
            "#4565ad"])
    plt.title(f"{x} and {y} vs {hue}")
    plt.show()


def plot_two_pie_plots(data, col1, col2, desc1, desc2, title):
    f, ax = plt.subplots(1, 2, figsize=(18, 7))

    vals = data[data[col1]][col2].value_counts()
    vals = dict(sorted(vals.items()))

    ax[0].pie(
        list(vals.values()),
        labels=list(vals.keys()),
        explode=[0, 0.1],
        autopct="%1.1f%%",
        colors=["#376f8c", "#1f4559"],
    )
    ax[0].set_title(f"{col2} distribution of {desc1}")

    data[data[col1] == False][col2].value_counts().plot.pie(
        explode=[0, 0.1], autopct="%1.1f%%", ax=ax[1], colors=["#a17f42", "#785e2e"]
    )
    ax[1].set_title(f"{col2} distribution of {desc2}")

    f.suptitle(title)
    plt.show()


def plot_pie_and_bar(data, col, desc, colors):
    f, ax = plt.subplots(1, 2, figsize=(18, 7))
    vals = data[col].value_counts()

    vals.plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], colors=colors)

    ax[1].bar(list(vals.keys()), list(vals), color=colors)
    ax[1].grid(False)

    f.suptitle(f"Distribution of people {desc}")
    plt.show()


def plot_roc_curve(model, X_train, Y_train, X_test, Y_test, classifier_name):
    _, ax = plt.subplots(1, 1, figsize=(15, 8))

    y_pred_proba = model.predict_proba(X_train)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(Y_train, y_pred_proba)

    plt.plot(fpr, tpr)
    print(f"AUC for training data {metrics.auc(fpr, tpr)}")

    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    print(f"AUC for testing data {metrics.auc(fpr, tpr)}")

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    ax.set_title(f"ROC Curve for {classifier_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.show()


def plot_model_comparison(result_comparison):
    plt.subplots(1, 1, figsize=(15, 7))

    x = np.arange(4)
    y1 = list(result_comparison["Accuracy"])
    y2 = list(result_comparison["Precision"])
    y3 = list(result_comparison["Accuracy"])
    width = 0.2

    plt.bar(x - 0.2, y1, width, color="#5868d1")
    plt.bar(x, y2, width, color="#d15896")
    plt.bar(x + 0.2, y3, width, color="#86d15e")

    plt.xticks(x, list(result_comparison.index))
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.legend(["Accuracy", "Precision", "Recall"])
    plt.title("Accuracy vs Precision vs Recall for different models")
    plt.show()


def plot_conf_matrix(model, X_train, Y_train, model_name, ax):
    y_pred = cross_val_predict(model, X_train, Y_train, cv=10)
    sns.heatmap(
        confusion_matrix(
            Y_train,
            y_pred),
        ax=ax,
        annot=True,
        fmt="2.0f")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    ax.set_title(f"Matrix for {model_name}")
    return y_pred


def plot_confusion_matrix_and_return_results(
    model1, model2, model3, model4, X_test, Y_test
):
    result_comparison = pd.DataFrame()

    f, ax = plt.subplots(2, 2, figsize=(15, 12))

    y_pred = cross_val_predict(model1, X_test, Y_test, cv=10)
    df = pd.DataFrame(
        {
            "Accuracy": metrics.accuracy_score(Y_test, y_pred),
            "Precision": metrics.precision_score(Y_test, y_pred),
            "Recall": metrics.recall_score(Y_test, y_pred),
        },
        index=["Logistic Regression"],
    )
    result_comparison = pd.concat([result_comparison, df])
    sns.heatmap(confusion_matrix(Y_test, y_pred),
                ax=ax[0, 0], annot=True, fmt="2.0f")
    ax[0, 0].set_title("Matrix for Logistic Regression")
    ax[0, 0].set_xlabel("Predicted labels")
    ax[0, 0].set_ylabel("True labels")

    y_pred = cross_val_predict(model2, X_test, Y_test, cv=10)
    df = pd.DataFrame(
        {
            "Accuracy": metrics.accuracy_score(Y_test, y_pred),
            "Precision": metrics.precision_score(Y_test, y_pred),
            "Recall": metrics.recall_score(Y_test, y_pred),
        },
        index=["KNN"],
    )
    result_comparison = pd.concat([result_comparison, df])
    sns.heatmap(confusion_matrix(Y_test, y_pred),
                ax=ax[0, 1], annot=True, fmt="2.0f")
    ax[0, 1].set_title("Matrix for KNN")
    ax[0, 1].set_xlabel("Predicted labels")
    ax[0, 1].set_ylabel("True labels")

    y_pred = cross_val_predict(model3, X_test, Y_test, cv=10)
    df = pd.DataFrame(
        {
            "Accuracy": metrics.accuracy_score(Y_test, y_pred),
            "Precision": metrics.precision_score(Y_test, y_pred),
            "Recall": metrics.recall_score(Y_test, y_pred),
        },
        index=["SVM"],
    )
    result_comparison = pd.concat([result_comparison, df])
    sns.heatmap(confusion_matrix(Y_test, y_pred),
                ax=ax[1, 0], annot=True, fmt="2.0f")
    ax[1, 0].set_title("Matrix for SVM")
    ax[1, 0].set_xlabel("Predicted labels")
    ax[1, 0].set_ylabel("True labels")

    y_pred = cross_val_predict(model4, X_test, Y_test, cv=10)
    df = pd.DataFrame(
        {
            "Accuracy": metrics.accuracy_score(Y_test, y_pred),
            "Precision": metrics.precision_score(Y_test, y_pred),
            "Recall": metrics.recall_score(Y_test, y_pred),
        },
        index=["Random Forest"],
    )
    result_comparison = pd.concat([result_comparison, df])
    sns.heatmap(confusion_matrix(Y_test, y_pred),
                ax=ax[1, 1], annot=True, fmt="2.0f")
    ax[1, 1].set_title("Matrix for Random-Forests")
    ax[1, 1].set_xlabel("Predicted labels")
    ax[1, 1].set_ylabel("True labels")

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
    return result_comparison


def plot_bar_plot(data):
    plt.subplots(1, 1, figsize=(15, 8))
    data_dict = dict(sorted(data.items(), key=lambda item: item[1]))
    plt.bar(
        range(
            len(data_dict)),
        list(
            data_dict.values()),
        align="center",
        color="#508f7b")
    plt.xticks(range(len(data_dict)), list(data_dict.keys()))
    plt.xticks(rotation=45)
    plt.title("Average CV Mean Accuracy")
    plt.xlabel("Model Name")
    plt.ylabel("Accuracy")
    plt.show()


def plot_feature_correlation_with(df, col):
    cols = [c for c in df.columns if c != col]
    corr_with_TravelInsurance = df[cols].corrwith(df[col])
    corr_with_TravelInsurance = corr_with_TravelInsurance.sort_values()

    plt.figure(figsize=(15, 7))
    corr_with_TravelInsurance.plot(kind="bar", color="#c2704c")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.title(f"Correlation with {col}", fontsize=15)
    plt.show()


def plot_heatmap(df, cols):
    corr = df[cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        mask=np.triu(corr),
        cmap="RdYlGn",
        annot_kws={
            "fontsize": 8})
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def plot_kde_plot(data1, data2, col, desc1, desc2, description):
    plt.figure(figsize=(15, 8))
    sns.kdeplot(
        data=data1,
        x=col,
        fill=True,
        common_norm=False,
        color="#599e5b",
        alpha=0.5,
        linewidth=0,
        label=desc1,
    )
    sns.kdeplot(
        data=data2,
        x=col,
        fill=True,
        common_norm=False,
        color="#507db5",
        alpha=0.5,
        linewidth=0,
        label=desc2,
    )
    plt.legend(loc="upper right")
    plt.title(f"Distribution of {description}", fontsize=13)
    plt.show()


def plot_pie_plot_circle(data, desc):
    vals = dict(data.value_counts())
    plt.figure(figsize=(16, 7))

    plt.pie(
        list(
            vals.values()),
        labels=list(
            vals.keys()),
        autopct="%1.1f%%",
        startangle=90)
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.tight_layout()
    plt.title(f"Distribution of number of {desc}", fontsize=12)
    plt.legend()
    plt.show() 


def plot_two_histograms(data1, data2, col, desc1, desc2):
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    data1[col].plot.hist(ax=ax[0], bins=10, edgecolor="black", color="#ab4652")
    ax[0].set_title(desc1)
    ax[0].set_xlabel(col)
    data2[col].plot.hist(ax=ax[1], color="#4565ad", bins=10, edgecolor="black")
    ax[1].set_title(desc2)
    ax[1].set_xlabel(col)
    f.suptitle(
        f"{col} distribution of people who {desc1} and {desc2}",
        fontsize=15)
    plt.show()


def plot_confusion_matrix_and_print_results(
        model1_results,
        model2_results,
        Y_test,
        model_name):
    _, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.heatmap(
        confusion_matrix(
            Y_test,
            model1_results),
        ax=ax[0],
        annot=True,
        fmt="2.0f")
    ax[0].set_title(f"Matrix for {model_name}")
    ax[0].set_xlabel("Predicted labels")
    ax[0].set_ylabel("True labels")

    sns.heatmap(
        confusion_matrix(
            Y_test,
            model2_results),
        ax=ax[1],
        annot=True,
        fmt="2.0f")
    ax[1].set_title(f"Matrix for bagged {model_name}")
    ax[1].set_xlabel("Predicted labels")
    ax[1].set_ylabel("True labels")

    print(
        f"Accuracy of Final {model_name} (not bagged) {round(metrics.accuracy_score(Y_test, model1_results), 3)}"
    )
    print(
        f"Accuracy of Bagged {model_name} {round(metrics.accuracy_score(Y_test, model2_results), 3)}"
    )
    print(
        f"Recall of Final {model_name} (not bagged) {round(metrics.recall_score(Y_test, model1_results), 3)}"
    )
    print(
        f"Recall of Bagged {model_name} {round(metrics.recall_score(Y_test, model2_results), 3)}"
    )
    print(
        f"Precision of Final {model_name} (not bagged) {round(metrics.precision_score(Y_test, model1_results), 3)}"
    )
    print(
        f"Precision of Bagged {model_name} {round(metrics.precision_score(Y_test, model2_results), 3)}"
    )

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
