import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# doing following this
# - ดู Nan ก่อน
#     - ถ้า Nan เยอะ remove, drop
#     - check corr คือ replace some value แล้วถ้า high corr ก็ไม่ drop
# - ดู dup Col
# - drop low variance ก็คือค่า var = 0 คือค่าไม่กระจายเลย เช่น เลข 1 ซ้ำกันทั้ง col ก็ตัดทิ้ง

###

try:
  dataset_names = sns.get_dataset_names()
except Exception as e:
  st.error(f"Error retrieving dataset names: {e}")
  dataset_names = []  # Set an empty list as a fallback

# Find the index of 'titanic' in the list (handle potential absence)
titanic_index = None
if 'titanic' in dataset_names:
  titanic_index = dataset_names.index('titanic')


st.title("Feature selection only have X (data no label no ml for find importance feature)")
# Create the selectbox with 'titanic' as the default if available
ds = st.selectbox("select dataset from seaborn.get_dataset_names()", dataset_names, key="dataset_selectbox", index=titanic_index)
# ds = st.selectbox("Select Dataset, sns.get_dataset_names())

### drop target ทำให้เหมือนมีแค่ DATA (X) เท่านั้น
drop_target_name = st.selectbox("do you want to manual drop target (if have have)? doing for titanic only support", ['yes', 'no'], index=0)
if drop_target_name == "yes":
    df = sns.load_dataset(ds)
    st.write("drop ['survived', 'alive'] before | pretend to have data just X no target data")
    target_column_index = ['survived', 'alive']# ซึ่งมี 2 ค่า
    df = df.drop(target_column_index, axis=1)# df_without_target
else:
    df = sns.load_dataset(ds)
###
with st.expander(f"Check unique of each features ( have {len(df.columns)} feature )"):
    df_nunique = df.nunique()
    df_nunique.name = "Count of unique values"
    df_nunique

###
def plot_corr(df, title_txt):
    st.subheader(f"Corr {title_txt}")
    corr = df.corr(numeric_only=True)# ดึงตัวเลขมาทำ corr โดยใช้ เพียสัน
    fig = plt.figure() # numberical atl+f8 # ตั้งชื่อ ว่าจะมีอะไรเข้าไปใส่
    sns.heatmap(corr, annot=corr)
    fig
    # BEFORE DROP NAN, AFTER DROP NAN
    # Filter correlations within -1 to 1 (excluding self and 1.0)
    filtered_corr = corr.where(~np.tril(np.ones(corr.shape)).astype(bool)) \
        .where(lambda x: np.abs(x) < 1.0)

    # Get top n/2 of feature most positive and negative correlations
    n = 3
    top_n_pos_corr = filtered_corr.stack().sort_values(ascending=False).head(int(df.shape[1]/n)) #ปัดลง
    top_n_neg_corr = filtered_corr.stack().sort_values(ascending=True).head(int(df.shape[1]/n))

    st.write(f"Top n/{n} of feature Most Positive Correlations (excluding self and 1.0):")
    st.write(top_n_pos_corr.to_frame(name='Correlation').reset_index())

    st.write(f"Top n/{n} of feature Most Negative Correlations (excluding self and 1.0):")
    st.write(top_n_neg_corr.to_frame(name='Correlation').reset_index())


###
def count_null_and_rename(df):
  """
  Counts the number of unique values (excluding nulls) in each column
  and renames the Series to indicate the number of null values.
  """
  null_counts = df.isna().sum()  # Count null values in each column
  null_counts.name = "Count of Null Values"
  return null_counts

st.title("1: Check Nan & Correlation &")
st.title("2: drop duplicate column")
with st.expander("Check Nan & Correlation compare before after drop nan"):
    df_nuniqu = count_null_and_rename(df.copy())
    df_nuniqu

    # Create two columns using st.columns
    col1, col2 = st.columns(2)
    # Display content in each column
    with col1:
        plot_corr(df, "Before DROP NAN")
    with col2:
        df_dropna_reset_index = df.dropna().reset_index(drop=True)
        plot_corr(df_dropna_reset_index, "After DROP NAN")

def df_all_column_cat2num_then_drop_dup(df):
    # df.dtypes
    # # Select df text columns
    # df_obj = df.select_dtypes(include='object')
    # df_bool = df.select_dtypes(include=bool)
    # df_cat = df.select_dtypes(include='category')    # Combine results (optional)
    # df_text_like = pd.concat([df_obj, df_bool, df_cat], axis=1)
    # #
    # df_num = df.select_dtypes(include='number')
    # #

    ###---ทำให้ดูเฉยๆ คือ pd.getdummies จะใช้ apply ได้กับ column ที่เป็น text เท่านั้น
    # obj_cols = [col for col in df.select_dtypes(include='object').columns]
    # bool_cols = [col for col in df.select_dtypes(include=bool).columns]
    # cat_cols = [col for col in df.select_dtypes(include='category').columns]
    # text_cols = []
    # text_cols.extend(obj_cols)
    # text_cols.extend(bool_cols)
    # text_cols.extend(cat_cols)

    # num_cols = [col for col in df.select_dtypes(include='number').columns]
    #
    # _df = pd.get_dummies(df[num_cols], dtype=int)
    # return _df, num_cols
    ###---
    ## create fn to vis to easy to drop
    st.write("embark_town ตัวเต็ม == embarked คือตัวย่อ , pclass ตัวเลข == class กับตัวหนังสือ")
    df = df.drop(columns=['embark_town', 'class', 'adult_male']) #embark_town ตัวเต็ม == embarked คือตัวย่อ , pclass ตัวเลข == class กับตัวหนังสือ
    df # กรณีนี้ deck มัน แปลงจาก null เข้า get_dummies แล้วเป็น 0 แทนของทุกตัว

    df = pd.get_dummies(df, dtype=int)
    st.write('drop อีกรอบ สำหรับตัวที่เป็น dummies variable problem')
    df = df.drop(columns=['sex_female', 'sex_male', 'who_woman'])#ไม่เอาตัว dup 'sex_female', 'sex_female' เพราะใช้ who แทนได้ ซึ่งได้เรื่องของ เด็ก กะผู้ใหญ่ด้วย
    df
    # st.write(len(df.columns))

    return df
    # df = _df.drop(columns=['sex_female'])
    # df = df.drop(columns=['pclass', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'])


with st.expander("before using pd.get_dummies you must to drop dub column first by manual"):
    df_int = df_all_column_cat2num_then_drop_dup(df)
    df_int_n = count_null_and_rename(df_int)
    # df_int_n# age หาย

    # df_int
    # Create two columns using st.columns
    col1, col2 = st.columns(2)
    # Display content in each column
    with col1:
        plot_corr(df, "BEFORE DROP NAN")
    with col2:
        st.write("Change all (cat,boolean,object) feature into numerical types then replace with stat")
        # df_int.columns
        df_dropna_reset_index = df_int.dropna().reset_index(drop=True)
        plot_corr(df_dropna_reset_index, "After DROP NAN")#just age # ถ้าทำแบบนี้ คือได้ถาม เทียบเท่ากับด้านบน

def fill_na_with_mode_and_type(df):
    """
    Fills NA values in each column of a pandas DataFrame with the mode,
    preserving data type (int or float). Prints column data types before
    and after filling.

    Args:
        df (pd.DataFrame): The DataFrame to fill NA values in.

    Returns:
        pd.DataFrame: The DataFrame with NA values filled with the mode.
    """
    # Iterate through each column
    for col in df.columns:
        # Print data type before filling
        print(f"Column '{col}' dtype before: {df[col].dtype}")

        # Check for non-NA values
        if df[col].dropna().size > 0:
            # Get the mode value (considering data type)
            if pd.api.types.is_numeric_dtype(df[col]):
                # If numeric, use mode for numeric type (mean for both int and float)
                mode_value = df[col].mode().iloc[0]
            else:
                # If not numeric, use the most frequent value (string)
                mode_value = df[col].mode().iloc[0]

            # Fill NA values with the mode
            df.loc[df[col].isna(), col] = mode_value

        # Print data type after filling (if mode was found)
        if mode_value is not None:
            pass
            # print(f"Mode for column '{col}': {mode_value}")
            # print(f"Column '{col}' dtype after: {df[col].dtype}")

    return df


import warnings

def fill_na_with_mean_and_type(df):
  """Fills NA values in each column of a pandas DataFrame with the mean, casting non-numeric values to integer before filling, preserving data type (int or float) when possible. Prints column data types before and after filling.

  Args:
      df (pd.DataFrame): The DataFrame to fill NA values in.

  Returns:
      pd.DataFrame: The DataFrame with NA values filled with the mean (or integer conversion for non-numeric).
  """
  # Iterate through each column
  for col in df.columns:
    # Print data type before filling
    print(f"Column '{col}' dtype before: {df[col].dtype}")

    # Check for non-NA values
    if df[col].dropna().size > 0:
      # Suppress warnings about casting during filling (optional)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)

      try:
        # Get the mean value (considering data type)
        mean_value = df[col].mean()

        # Check if numeric before casting
        if not pd.api.types.is_numeric_dtype(df[col]):
          # Cast to integer for non-numeric columns (may lose precision)
          mean_value = int(mean_value)
          print(f"Warning: Casting non-numeric column '{col}' to integer for filling.")
      except ValueError:
        # Handle potential exceptions during casting (e.g., non-numeric mean)
        print(f"Warning: Could not calculate mean or cast to integer for column '{col}'.")
        continue

      # Fill NA values with the mean (or casted integer)
      df.loc[df[col].isna(), col] = mean_value

      # Print data type after filling (if mean was calculated)
      # print(f"Fill value for column '{col}': {mean_value}")
      # print(f"Column '{col}' dtype after: {df[col].dtype}")
  return df


with st.expander("Try with replace nan with other stat"):
    # Create two columns using st.columns
    col1, col2, col3  = st.columns(3)
    # Display content in each column
    with col1:
        df_dropna_reset_index = df_int.dropna().reset_index(drop=True)
        plot_corr(df_dropna_reset_index, "After DROP NAN")
    with col2:
        st.write("mode")
        # print("-------------mode_c2-----------------")
        df_int_fill_mode = fill_na_with_mode_and_type(df_int)
        plot_corr(df_int_fill_mode, "After REPLACE NAN with MODE")
    with col3:
        st.write("means")
        # print("-------------means_c3----------------")
        df_int_fill_means = fill_na_with_mean_and_type(df_int)
        plot_corr(df_int_fill_means, "After REPLACE NAN with Means")


st.title("3: Finding and dropping low-variance features in df")
with (st.expander("เข้าใจง่ายๆคือ column ที่ ค่าไม่มีความหลากหลาย เพราะถ้าเอาเข้า model มันจะทำให้ model ซับซ้อนโดยเปล่าประโยชน์ ทั้งนี้จะรู้ได้ยังไง ก็ขึ้นกับการตีความข้อมูลที่มี และ target ที่จะหา")):
    st.write("before selection feature", df_dropna_reset_index.columns)

    p = 0.25  # เช่น กรองสแปมอีเมล | การกำหนด 0.25 คืออีเมลใดๆ ที่มีความน่าจะเป็นสแปมมากกว่า 25% จะถูกทำเครื่องหมายว่าเป็นสแปม
    th = p * (1 - p)
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=th)
    selector.fit(df_dropna_reset_index)
    # st.write("_________")
    # st.write(selector.variances_)
    # X_ = selector.transform(df_dropna_reset_index)
    # # st.write(X_)
    # # st.write("_________")
    st.write(selector.get_feature_names_out().tolist())# last feature more than th
    selector.get_feature_names_out().tolist()
    st.write("after selection feature")
    df_dropna_reset_index[selector.get_feature_names_out().tolist()]