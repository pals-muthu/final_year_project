
from pathlib import Path


def put_to_csv(base_path, df, file_name=None):
    if not file_name:
        file_path = Path(base_path).resolve().joinpath('temp.csv')
    else:
        file_path = Path(base_path).resolve().joinpath(file_name)
    df.to_csv(file_path, sep='\t', encoding='utf-8')
    print("#Done adding to file: ", file_path)
