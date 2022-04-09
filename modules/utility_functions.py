import pandas as pd


def update_nan_most_frequent_category_tuple(DataFrame, src_df, ColName):

    temp_df = src_df[ColName].apply(
        lambda x: x[0] if type(x) == tuple else '')

    # for index, row in DataFrame.iterrows():
    #     print("row: ", row[ColName], type(row[ColName]))
    # print("temp_df: ", temp_df)

    # .mode()[0] - gives first category name
    most_frequent_category = temp_df.mode()[0]
    # print("most frequent: ", most_frequent_category)

    # replace nan values with most occured category
    # DataFrame[ColName] = DataFrame[ColName].apply(lambda x: (
    #     '{}'.format(most_frequent_category),) if pd.isnull(x) else x)
    DataFrame[ColName] = DataFrame[ColName].apply(lambda x: (
        '{}'.format(0),) if pd.isnull(x) else x)


def update_nan_most_frequent_category(DataFrame, src_df, ColName):

    # replace nan values with most occured category
    # DataFrame[ColName] = DataFrame[ColName].apply(lambda x: (
    #     '{}'.format(most_frequent_category),) if pd.isnull(x) else x)
    DataFrame[ColName] = DataFrame[ColName].apply(
        lambda x: 0 if pd.isnull(x) else x)


def remove_nan(DataFrame, ColName):
    # DataFrame = DataFrame.dropna(subset=[ColName])
    DataFrame = DataFrame[DataFrame[ColName].notnull()]
