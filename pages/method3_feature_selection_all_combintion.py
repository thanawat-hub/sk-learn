import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from itertools import combinations, chain

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def EFS(estimator, X, Y, cv=5, verbose=False):
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(range(n_features), i) for i in range(1, 5))
    best_score = -np.inf
    best_subset = None
    scores = []  # List to store scores and subsets

    for i, subset in enumerate(subsets):
        subset = list(subset)
        score = cross_val_score(estimator, X.iloc[:, subset], Y, cv=cv).mean()
        scores.append((tuple(subset), score))  # Store tuple of subset and score

        if score > best_score:
            best_score = score
            best_subset = subset

        if verbose:
            print(i, score, subset)

    return X.columns[best_subset], scores

########### all data
# with st.expander("Load dataset"):
    # try:
    #   dataset_names = sns.get_dataset_names()
    # except Exception as e:
    #   st.error(f"Error retrieving dataset names: {e}")
    #   dataset_names = []  # Set an empty list as a fallback
    #
    # titanic_index = None
    # if 'titanic' in dataset_names:
    #   titanic_index = dataset_names.index('titanic')
    #
    # ds = st.selectbox("from seaborn.get_dataset_names()", dataset_names, key="dataset_selectbox", index=titanic_index)
    # df = sns.load_dataset(ds)

    # df_X = df.drop(columns=['survived', 'alive'])
    # st.write("Data (X)")
    # df_X
    # df_Y = df['alive']
    # st.write("Target (Y)")
    # df_Y

# n_features = df_X.shape[1]
# st.write(n_features)#13

# ### to test to see all case
# c=0
# for i in range(1, n_features):
#     for x in combinations(list(range(n_features)), i): # เริ่มที่ 0 จบ จน จำนวน feature ทุกคู่อับดันที่เป็นไปได้
#         print(x)
#         c=c+1
# print(c)# 2^ 13
# ###
# end load datasets from sns
###########

########### try just one datasets low dimension
with st.expander("Load dataset iris"):

    X = load_iris(as_frame=True)
    n_features = X.data.shape[1]

### to use in chain
chain.from_iterable(combinations(range(n_features), i) for i in range(1, n_features))

clf = KNeighborsClassifier(n_neighbors=3)
# selected

# print('=-----------------------=')
# print(type(selected))

# Streamlit app
st.title("Exhaustive Feature Selection with Streamlit")

if st.button("Run Feature Selection"):
    st.write("Running exhaustive feature selection...")
    selected_features, scores = EFS(clf, X.data, X.target, cv=StratifiedKFold(5), verbose=True)

    # Prepare data for plotting
    subsets, score_values = zip(*scores)
    subsets = ['-'.join(map(str, subset)) for subset in subsets]  # Convert tuple to string for labeling

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(subsets, score_values, marker='o')
    plt.xticks(rotation=90)
    plt.xlabel('Feature Subsets')
    plt.ylabel('Cross-Validation Score')
    plt.title('Exhaustive Feature Selection Scores')
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(fig)

    st.write("Best feature subset:")
    st.write(selected_features)