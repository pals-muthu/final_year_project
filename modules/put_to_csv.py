
from pathlib import Path
import pandas as pd


def put_to_csv(base_path, df, file_name=None):
    if not file_name:
        file_path = Path(base_path).resolve().joinpath('temp.csv')
    else:
        file_path = Path(base_path).resolve().joinpath(file_name)
    df.to_csv(file_path, sep='\t', encoding='utf-8')
    print("#Done adding to file: ", file_path)


def put_np_array_to_csv(base_path, np_array, file_name=None):
    df = pd.DataFrame(data=np_array[1:, 0:],    # values
                      columns=np_array[0, 0:])
    put_to_csv(base_path, df, file_name)


def put_unconstructed_np_array_to_csv(base_path, np_array, file_name=None):
    df = pd.DataFrame(data=np_array)
    put_to_csv(base_path, df, file_name)
