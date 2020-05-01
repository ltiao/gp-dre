# -*- coding: utf-8 -*-
"""
Scikit-learn Datasets
=====================
"""
# sphinx_gallery_thumbnail_number = 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                              load_diabetes, load_boston)


def make_data(dataset):

    data = pd.DataFrame(dataset.data, columns=dataset.feature_names) \
             .assign(target=dataset.target)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    return data

# %%
# Classification datasets
# -----------------------
#
# Iris dataset
# ++++++++++++


iris = load_iris()

g = sns.pairplot(data=make_data(iris), hue="target",
                 palette="colorblind", corner=True)


# %%
# Wine dataset
# ++++++++++++


wine = load_wine()

g = sns.pairplot(data=make_data(wine), hue="target",
                 palette="colorblind", corner=True)

# %%
# Breast cancer dataset
# +++++++++++++++++++++

breast_cancer = load_breast_cancer()

g = sns.pairplot(data=make_data(breast_cancer), hue="target",
                 palette="colorblind", corner=True)

# %%
# Regression datasets
# -------------------
#
# Boston housing dataset
# ++++++++++++++++++++++

boston = load_boston()

g = sns.PairGrid(data=make_data(boston), hue="target",
                 palette="viridis", corner=True)
g.map_lower(plt.scatter)

# %%
# Diabetes dataset
# ++++++++++++++++

diabetes = load_diabetes()

g = sns.PairGrid(data=make_data(diabetes), hue="target",
                 palette="viridis", corner=True)
g.map_lower(plt.scatter)
