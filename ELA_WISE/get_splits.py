import pandas as pd
import json

def get_files_from_split(split):
    """ "
    Get filenames for real and fake samples

    Parameters
    ----------
    split : pandas.DataFrame
        DataFrame containing filenames
    """
    files_1 = split[0].astype(str).str.cat(split[1].astype(str), sep="_")
    files_2 = split[1].astype(str).str.cat(split[0].astype(str), sep="_")
    files_real = pd.concat([split[0].astype(str), split[1].astype(str)]).to_list()
    files_fake = pd.concat([files_1, files_2]).to_list()
    return files_real, files_fake


if __name__=="__main__":
    test_split = pd.read_json('train.json', dtype=False)
    test_files_real, test_files_fake = get_files_from_split(test_split)
    print("test_files_real::",test_files_real)
    print("test_files_fake::",test_files_fake)
