import read_and_validate.read_cityscapes as rc
import read_and_validate.validate_cityscapes as vc
import analysis.graphics as gr
import analysis.other as oth
import write.write as wr

def main():
    """
    Funció principal de l'aplicació.
    :return:
    """
    print("\nExercici 1")
    print("Llegint dades...")
    data = rc.read_cityscapes()

    print("Exercici 1.2")
    print(data.loc[1:10].to_string())
    print("""
        En el cas que tinguéssim milers d'arxius amb més d'un Gb per
        arxiu no és raonable carregar el dataset sencer en memòria.
        Una possible estratègia per fer aixó podria consistir en implementar
        un mòdul format per una funció de lectura que llegeixi una porció
        de mida raonable de les dades i un compilador que integri els 
        resultats parcials que anem obtenint de cadascuna de les porcions.
        Si per exemple vulguéssim implementar una funció que compti el número
        d'objectes hi ha en cadascuna de les imatges, els passos a seguir serien 
        els següents:
            1. Carregar una porció de les dades en memòria.
            2. Comptar el número d'objectes en les imatges que hem carregat.
            3. Compilar la informació que hem obtingut amb la que que
                hem obtingut de les porcions anteriors.
            4. Si hi ha més dades per processar tornem al pas 1, sinó hem acabat.
    """)

    print("\nExercici 2")
    error_rows = vc.validate_YOLO(data.iloc[:, 2:])
    if len(error_rows) == 0:
        print("No hi ha línies errònies\n")
    else:
        print("Hi ha errors en els fitxers següents:")
        print(data.loc[error_rows, "filename"].drop_duplicates().to_string())
    # eliminem files incorrectes
    vc.remove_filenames(data, error_rows)

    print("\nExercici 3")
    vc.validate_images(data)

    print("\nExercici 4")
    data = data[data["confidence"] > 0.4]

    print("\nExercici 4.1")
    gr.plot_object_histogram(data)

    print("\nExercici 4.2")
    gr.plot_top_objects_by_img_distribution(data)

    print("\nExercici 4.3")
    num_objectes_per_imatge = data.groupby("filename")["filename"].value_counts().mean()
    print("El nombre total d'objectes per imatge és {:.2f}.".format(num_objectes_per_imatge))

    print("\nExercici 4.4")
    print(oth.compute_top3_dict(data))

    print("\nExercici 5")
    gr.plot_cars_by_year_and_city(data)

    print("\nExercici 6")
    print("""
        La solució que proposem consisteix en obtenir la distribució de cada 
        objecte en la ciutat i considerar el diagrama de caixa (Box-plot)
        d'aquesta distribució. Considerarem com a fotos intruses aquelles en les
        que un hi hagi objectes en proporcions atípiques, concretament, fotos en les
        que hi hagi objectes en quantitats fora del rang que determina el box-plot: 
        mediana +- 1.5 x rang interquartíl·lic.
    """)
    wrong_filenames = oth.find_outliers_zurich(data)
    print("Els fitxers que s'han considerat erronis són els següents:\n")
    print("\n".join(wrong_filenames))

    print("\nExercici 7")
    wr.write_working_data(data, wrong_filenames)


main()
