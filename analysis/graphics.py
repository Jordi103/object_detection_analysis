import matplotlib.pyplot as plt


def plot_object_histogram(data):
    """
    Mostra un gràfic de barres del número de cops que ha aparegut cada objecte.
    :param data: dataframe en format com el retornat per read_cityscapes
    :return:
    """
    # series amb núm objecte a l'índex, count com a valor
    counts = data["object"].value_counts().sort_values(ascending=False)
    x, y = counts.index, counts.values
    print("Els 5 objectes amb més ocurrències són")
    print("Objecte\tNúm. ocurrències")
    for i in range(5):
        print(x[i], '\t\t', y[i])
    plt.bar(x, y)
    plt.show()


def plot_top_objects_by_img_distribution(data):
    """
    Mostra l'histograma del número d'instàncies de cada objectre en cada imatge;
    considerem només els 5 objectes més populars.
    :param data: dataframe en format com el retornat per read_cityscapes
    :return:
    """
    # els 5 objectes
    objects = data["object"].value_counts().sort_values(ascending=False).index[0:5]
    # rows in which the 5 objects appear
    df = data[data["object"].isin(objects)][["filename", "object"]]
    df = df.groupby("filename")["object"].value_counts()
    df.name = "counts"
    df = df.reset_index()
    df["counts"].hist(by=df["object"], bins=20, density=True)
    plt.show()


def plot_cars_by_year_and_city(data):
    """
    Mostra un gràfic de barres del número de cotxers per any i ciutat.
    :param data: dataframe en format retornat per read_cityscapes.
    :return:
    """
    # seleccionem les files corresponents a cotxes
    relevant_data = data[data["object"] == 2][["city"]].copy()
    relevant_data["year"] = data["filename"].apply(lambda x: x.split('-')[-1])
    relevant_data.groupby("year").value_counts().unstack().plot.bar()
    plt.show()


if __name__ == '__main__':
    from read_and_validate.read_cityscapes import read_clean_sample
    data = read_clean_sample(n=-1, labels_path='../dataset_cities/labels/')
    plot_top_objects_by_img_distribution(data)

