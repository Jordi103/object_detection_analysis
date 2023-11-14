Aquest mòdul de python anomenat pac4 té l'objectiu de proporcionar un seguit
d'eines per la lectura, visualització i anàlisi del dataset cityscapes (més
informació a https://www.cityscapes-dataset.com). 

Per tal d'executar el programa el dataset ha d'estar contingut en el diectori
dataset_cities i ha de contenir els fitxers següents:
    1. Directori amb les imatges.
    2. Directori amb fitxers YOLO sobre les imatges. Els fitxers YOLO són fitxers
        en els que s'especifica els objectes detectats en cada imatge amb l'algorisme
        YOLO. Els fitxers contenen files amb sis ítems relacionats amb l'objecte
        detectat:
            1. El codi del tipus d'objecte detectat.
            2. La coordenada x del centre del rectangle que envolta l'objecte.
            3. La coordenada y del centre del rectangle que envolta l'objecte.
            4. L'amplada del rectangle que envolta l'objecte.
            5. L'alçada del rectangle que envolta l'objecte.
            6. El grau de confiança amb la que l'algorisme YOLO ha detectat l'objecte
        El codi d'objecte és un número enter entre 0 i 80 i la resta de números són
        decimals entre 0 i 1.
    3. El significat de cada codi d'objecte.

EXECUCIÓ DEL CODI
Les diferents funcionalitats demanades en l'enunciat de la PAC s'executen de manera
seqüencial des del fitxer python main.py de la carpeta principal del projecte. Per
executar aquest fitxer podem o bé carregar el projecte des d'un editor com pycharm
o bé directament des de la línia de comandes. Per fer-ho des de la línia de comandes
obrim una terminal dins la carpeta principal del projecte i executem

    $ source venv/bin/activate
    $ python3 main.py


ESTRUCTURA DEL CODI
El codi està format en tres paquets principals i un paquet addicional de tests.
L'estructura del codi és la següent

    + read_and_validate: mòduls per llegir i validar les dades.
        + read_cityscapes: funcions per llegir les dades.
            - read_cityscapes: llegeix les dades referents al dataset.
            - read_cityscapes_sample: llegeix les dades d'una part del dataset.
            - read_clean_sample: llegeix i neteja dades d'una part del dataset.
        + validate_cityscapes: mòduls per validar alguns aspectes de les dades llegides
                               read_cityscapes.
            - validate_yolo: comprova que un conjunt de dades segueix el format YOLO.
            - remove_filenames: elimina totes les files del dataframe com primer paràmetre
                que provenen del mateix fitxer que les files passades com a segon paràmetre.
            - validate_images: mostra per pantalla la primera foto de cada ciutat (en ordre
                lexicogràfic) amb els rectangles descrits en el fitxer YOLO de la foto.

    + analysis: mòduls per analitzar les dades
        + graphics: funcions per mostrar representacions gràfiques del dataset.
            - plot_object_distribution: Mostra un gràfic de barres del número de cops que ha
                aparegut cada objecte.
            - plot_top_objects_by_img_distribution: Mostra l'histograma del número d'instàncies
                de cada objectre en cada imatge; considerem només els 5 objectes més populars.
            - plot_cars_by_year_and_city: Mostra un gràfic de barres del número de cotxers per any i ciutat.
        + other: altres funcions per analitzar les dades.
        - extract_top3: Extreu els objectes en els tres objectes més representats en el fitxer del paràmetre.
            - compute_top3_dict: Calcula, per cada objecte, el número de cops que l'objecte ha estat
                 en el top 3 d'objectes variant imatges.
            - find_outliers_zurich: Trobem les imatges infiltrades basant-nos en la distribució la presència
                dels objectes en les imatges.
    + write: mòduls per escriure dades en fitxers.
        + write: funcions per escriure dades en fitxers.
            - write_working_data: Escriu per cada fitxer del dataset el nom de la imatge, el número de cotxes,
                el número de semàfors, el número de persones, ciutat, any i si és o no una
                imatge detectada com a intrusa.

TESTS
Els tests comproven les funcionalitats de l'aplicació que no consisteixen en llegir dades,
escriure dades o mostrar gràfics per pantalla. Es poden executar amb les comandes següents:

    $ source venv/bin/activate
    $ python3 -m unittest

LLICÈNCIA
Aquest programa es publica sota la llicència GPLv3.0.
