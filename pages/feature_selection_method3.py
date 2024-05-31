import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns

from itertools import combinations, chain

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def EFS(estimator, X, Y, cv=5, verbose=False): #Exhaustive feature selection
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(range(n_features), i) for i in range(1, 5))
    best_score = -np.inf
    best_subset = None#คงไม่น้อยกว่า อนัน อยู่แล้ว
    for i, subset in enumerate(subsets):
        subset = list(subset)
        score = cross_val_score(estimator, X.iloc[:, subset], Y, cv=cv).mean() # ถ้า score ใหม่ดีกว่า
        # เก็บเป็นค่าใหม่ที่ดีสุด
        if score > best_score:
            best_score = score
            best_subset = subset
        if verbose:
            print(i, score, subset)
    return X.columns[best_subset]


# n_features = X.data.shape[1]
# chain.from_iterable(combinations(range(n_features), i) for i in range(1, 5))

with st.expander("Load dataset"):
    try:
      dataset_names = sns.get_dataset_names()
    except Exception as e:
      st.error(f"Error retrieving dataset names: {e}")
      dataset_names = []  # Set an empty list as a fallback

    # Find the index of 'titanic' in the list (handle potential absence)
    titanic_index = None
    if 'titanic' in dataset_names:
      titanic_index = dataset_names.index('titanic')


    # Create the selectbox with 'titanic' as the default if available
    ds = st.selectbox("from seaborn.get_dataset_names()", dataset_names, key="dataset_selectbox", index=titanic_index)

    df = sns.load_dataset(ds)


clf = KNeighborsClassifier(n_neighbors=3)
selected = EFS(clf, X.data, X.target, cv=StratifiedKFold(5), verbose=True)
print(selected)