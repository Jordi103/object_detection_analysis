from collections import OrderedDict
import numpy as np


def extract_top3(data, filename):
    """
    Extreu els objectes en els tres objectes més representats en el fitxer del paràmetre.
    :param filename: nom del fitxer
    :return: tupla dels tres objectes en el top 3
    """
    objects_df = data[data["filename"] == filename]["object"].value_counts().sort_values(ascending=False)
    objects_df = objects_df.reset_index()
    objects_df.rename(columns={"object": "counts", "index": "object"}, inplace=True)

    # menys de 3 objectes
    if len(objects_df) <= 3:
        return objects_df["object"].to_list()
    mf = objects_df.loc[0, "counts"]
    mf_objects = objects_df[objects_df["counts"] == mf]
    num_mf_objects = len(mf_objects["object"])
    # més de 3 objectes amb màxima freqüència
    if num_mf_objects >= 3:
        return mf_objects["object"].to_list()
    counts_mf_3 = objects_df["counts"].iloc[2]
    counts_mf_4 = objects_df["counts"].iloc[3]
    # tres primers valors iguals, 4t diferent
    if len(objects_df["counts"].iloc[0:3].unique()) == 1:
        if counts_mf_3 != counts_mf_4:
            return objects_df["object"].iloc[0:3].to_list()
    # dos primers valors mateixa freqüència, 3r diferent
    if len(objects_df["counts"].iloc[0:2].unique()) == 1:
        if counts_mf_4 == counts_mf_3:
            return objects_df["object"].iloc[0:2].to_list()
        else:
            return objects_df["object"].iloc[0:3].to_list()
    if len(objects_df["counts"].iloc[1:4].unique()) == 1:
        return [objects_df["object"].iloc[0]]
    return objects_df["object"].iloc[0:3].to_list()


def compute_top3_dict(data):
    """
    Calcula, per cada objecte, el número de cops que l'objecte ha estat
    en el top 3 d'objectes variant imatges
    :param data: pandas dataframe en el format de read_cityscapes
    :return: diccionari
    """
    # inicialitzem el diccionari
    times_in_top3 = {o: 0 for o in data["object"].unique()}
    for file in data["filename"].unique():
        top3 = extract_top3(data, file)
        for o in top3:
            times_in_top3[o] += 1
    return OrderedDict(sorted(times_in_top3.items(), key=lambda x: x[1], reverse=True))


def find_outliers_zurich(data):
    """
    Trobem les imatges infiltrades basant-nos en la distribució la presència
    dels objectes en les imatges. Considerem com a intruses les que cauen fora
    del "bigots" d'un box-plot.
    :param data: pandas dataframe en el format retornat per read_cityscapes
    :return:
    """
    relevant_data = data[data["city"] == "zurich"]
    df_counts = relevant_data.groupby("object")["filename"].value_counts().reset_index(0)
    df_counts.rename(columns={"filename": "counts"}, inplace=True)
    df_counts.reset_index(inplace=True)

    incorrect_files = []
    for obj in df_counts["object"].unique():
        distr = df_counts[df_counts["object"] == obj]["counts"].values
        median = np.median(distr)
        q1 = np.quantile(distr, 0.25)
        q3 = np.quantile(distr, 0.75)
        ric = q3 - q1
        accept_range = (median - 1.5*ric, median + 1.5*ric)

        result = df_counts[df_counts["object"] == obj]
        result = result[(result["counts"] < accept_range[0]) | (result["counts"] > accept_range[1])]
        incorrect_files += result["filename"].to_list()
    return incorrect_files


if __name__ == '__main__':
    from read_and_validate.read_cityscapes import read_clean_sample
    data = read_clean_sample(n=-1, labels_path='../dataset_cities/labels/')
    print(compute_top3_dict(data))

