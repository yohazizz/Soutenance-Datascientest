# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:59 2024

@author: azizb
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
# Chargement des données
@st.cache_data
def load_data():
    genome_scores = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/genome-scores.csv'
    genome_tags = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/genome-tags.csv'
    links = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/links.csv'
    movies = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/movies.csv'
    ratings = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/ratings.csv'
    tags = 'C:/Users/azizb/Desktop/Datatsets projet/ml-20m/ml-20m/tags.csv'

    gs = pd.read_csv(genome_scores)
    gt = pd.read_csv(genome_tags)
    links = pd.read_csv(links)
    movies = pd.read_csv(movies)
    ratings = pd.read_csv(ratings)
    tags = pd.read_csv(tags)

    return gs, gt, links, movies, ratings, tags

gs, gt, links, movies, ratings, tags = load_data()

# Fusion et transformation des données
df_g = pd.merge(gt, gs, on='tagId', how='inner')
df_g['movieId'] = df_g['movieId'].astype(str)
indices_max_relevance = df_g.groupby(['movieId'])['relevance'].idxmax()
df_g_result = df_g.loc[indices_max_relevance]
df_g_result['movieId'] = df_g_result['movieId'].astype(int)
df_movies = pd.merge(df_g_result, movies, on='movieId', how='right')

ratings_copy = ratings.copy()
nombre_entrees_a_conserver = int(len(ratings) / 10)
indices_a_conserver = np.random.choice(ratings.index, size=nombre_entrees_a_conserver, replace=False)
ratings_reduit = ratings_copy.loc[indices_a_conserver]
df = pd.merge(ratings_reduit, df_movies, on='movieId', how='left')
df = df.dropna()

unique_genres = ['Comedy', 'Action', 'Crime', 'Drama', 'Mystery', 'Adventure', 'Musical', 'Documentary', 'IMAX', 'Thriller', 'Sci-Fi', 'Fantasy', 'Horror', 'Romance', 'Animation', 'Children', 'War', 'Western', 'Film Noir', '(no genres listed)']
df['genres'] = df['genres'].str.split('|')
genre_columns = pd.DataFrame(df['genres'].apply(lambda x: [1 if genre in x else 0 for genre in unique_genres]).tolist(), columns=unique_genres, index=df.index)
df = pd.concat([df, genre_columns], axis=1)
df = df.drop('genres', axis=1)

# Configuration de la page Streamlit
st.sidebar.title("Sommaire")
pages = ["Introduction", "Récupération et mise en forme des données", "Visualisation", "Exploitation des données", "Présentation des différentes approches", "Modélisation et résultats", "Conclusion", "Projet de classification de films"]
page = st.sidebar.radio("Aller vers", pages)

if page == "Introduction":
    st.title("Rapport de projet")
    st.header("Création d’un système de Recommandation de film")
    st.markdown("""
    ## Introduction
    La finalité de ce projet vise à proposer un système de recommandation de film
    adapté aux formes actuelles d’évaluation des films et persistant dans le temps.
    Ce système est mis en place grâce à l’application de collaborative filtering.
    ...
    Les bases de données à disposition :
    - [Movielens](https://grouplens.org/datasets/movielens/20m/)
    - [IMDB](https://www.imdb.com/interfaces/)
    """)

elif page == "Récupération et mise en forme des données":
    st.title("Récupération et mise en forme des données")
    st.write("""
    ## Récupération et mise en forme des données
    - Description des étapes de chargement et de transformation des données.
    """)
    moviesData = pd.read_csv(r"C:/Users/azizb/Desktop/Datatsets projet/test_2 (1).csv",sep=',', index_col=0)
    moviesData.describe()
    # Sélection des colonnes numériques pour la matrice de corrélation
    numeric_columns = moviesData.select_dtypes(include='number')
    # Calcul la matrice de corrélation
    correlation_matrix = numeric_columns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Matrice de corrélation')
    sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='coolwarm')

    if page == "Récupération et mise en forme des données":

        st.subheader("Modélisation des données:")
        image =r"C:/Users/azizb/Desktop/Datatsets projet/UML.PNG"
        st.image(image, caption='Your Image Caption', use_column_width=True)
        st.subheader("Problématiques rencontrées:")
        st.write("""
           -Redondance au niveau des données.    


           -Taille des fichiers.


           -Taille de la mémoire (16GO).


           -types de données.
           """)

        st.subheader("Adaptation des données:")
        st.write("""
               -Transformation du type de données pour certaines colonnes:
                
                int64 => unit32


                float64 => unit32


                float64 => float32


                gain: 25% au niveau de la taille
                
               -Taille des fichiers:


                jusqu'a 90% pour certain fichier
                
               """)




        st.subheader("Matrice de corrélation")
        st.pyplot(fig)

        image = r"C:/Users/azizb/Desktop/Datatsets projet/data1.PNG"
        st.image(image, caption='Your Image Caption', use_column_width=True)

        image = r"C:/Users/azizb/Desktop/Datatsets projet/data2.PNG"
        st.image(image, caption='Your Image Caption', use_column_width=True)

        st.write(moviesData.shape)
        st.dataframe(moviesData.describe())
        st.subheader("Transformation des données:")
        st.write("""       
                      -Transformer les colonnes de type String:


                       le0 = LabelEncoder()


                       moviesData['tag_x'] = le0.fit_transform(moviesData['tag_x'])   ==> tag_x_label
                       
                       
                       le1 = LabelEncoder()


                       moviesData['tag_y'] = le1.fit_transform(moviesData['tag_y'])   ==> tag_y_label


                       le2 = LabelEncoder()


                       moviesData['relevance'] = le2.fit_transform(moviesData['relevance'])     ==> relevance_label




                      -suppression des colonnes avec une corrélation forte ou qui n'ont pas de pertinence dans le processus d'apprentissage:


                       moviesData = moviesData.drop(['movieId', 'tag_x',  'tag_y', 'relevance'], axis=1) 
                
                      """)

    #moviesData = moviesData.drop(['tagId','movieId','title', 'tag_x',  'tag_y', 'relevance', 'tmdbId','timestamp_x','timestamp_y'], axis=1)
    if page == "Le jeu de données":

        st.write("""
        
                      -Utilisation d'un object  RandomOverSampler() permet d'équilibrer la répartition de l'échantillon sur les différents labels
                     
                     
                      -Utilisation d'un object rUs = MinMaxScaler () permet de normaliser les données et les mettre a la méme échelle
                    
                    
                      DATA FOR MODEL:
                      
                        note                                                        y_train
                        
                        
                        300                                                        15027
                        
                        
                        700                                                        15027
                        
                        
                        600                                                        15027
                        
                        
                        800                                                        15027
                        
                        
                        900                                                        15027
                        
                        
                        500                                                        15027
                        
                        
                        400                                                        15027
                        
                        
                        200                                                        15027
                                            
                        
                        100                                                        15027
                        
                        
                        0                                                          15027
                        
                        
                        
                      """)

    print('DATA FOR MODEL:')
    X = moviesData.drop(['note'], axis=1)
    y = moviesData['note']
    y = y.apply(lambda x: x * 100)
    # Division des  données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Random Undersampling (équilibrer le jeux de donné)
    rUs = RandomOverSampler()
    X_train, y_train = rUs.fit_resample(X_train, y_train)
    print("y_train", y_train.value_counts())
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    print('Classes échantillon overSampled :', dict(pd.Series(y_train).value_counts()))
    # Normalisation les données
elif page == "Visualisation":
    st.title("Data Visualisation de Films")
    options = st.sidebar.radio("Choisir une visualisation", ['Distribution des genres', 'Distribution des tags', 'Distribution des notations', 'Genres les mieux notés'])

    # Fonction pour afficher les graphiques
    def plot_genre_distribution():
        image =r"C:/Users/azizb/Desktop/Datatsets projet/Image1.png"
        st.image(image, caption='Distribution des genres de films', use_column_width=True)

    def plot_tag_distribution():
        image =r"C:/Users/azizb/Desktop/Datatsets projet/Image2.png"
        st.image(image, caption='Distribution des tags de films', use_column_width=True)

    def plot_ratings_distribution():
        image =r"C:/Users/azizb/Desktop/Datatsets projet/Image3.png"
        st.image(image, caption='Distribution des notations', use_column_width=True)

    def plot_top_rated_genres():
        image =r"C:/Users/azizb/Desktop/Datatsets projet/Image4.png"
        st.image(image, caption='Genres les mieux notés en fonction de la moyenne des notes', use_column_width=True)

    # Affichage des graphiques en fonction de la sélection
    if options == 'Distribution des genres':
        plot_genre_distribution()
    elif options == 'Distribution des tags':
        plot_tag_distribution()
    elif options == 'Distribution des notations':
        plot_ratings_distribution()
    elif options == 'Genres les mieux notés':
        plot_top_rated_genres()

elif page == "Exploitation des données":
    st.title("Exploitation des données")
    st.write("## Exploitation des données")

elif page == "Présentation des différentes approches":
    st.title("Présentation des différentes approches")
    st.write("## Présentation des différentes approches")

elif page == "Modélisation et résultats":
    st.title("Modélisation et résultats")
    st.write("## Modélisation et résultats")

elif page == "Conclusion":
    st.title("Conclusion")
    st.write("## Conclusion")

elif page == "Projet de classification de films":
    st.title("Projet de classification de films")
    st.write("### Introduction")
    st.write("""
    Dans ce projet, on vous propose la mise en place d'un système de recommandation de film grâce à l’application de collaborative filtering.
    - Préparation du dataset qui sera utilisé par les différents modèles.
    - Test de plusieurs modèles et choix du modèle le plus adapté.
    """)

    # Chargement du jeu de données de classification
    moviesData = pd.read_csv("C:/Users/azizb/Desktop/Datatsets projet/test_2 (1).csv", sep=',', index_col=0)
    moviesData.describe()
    # Sélection des colonnes numériques pour la matrice de corrélation
    numeric_columns = moviesData.select_dtypes(include='number')
    # Calcul de la matrice de corrélation
    correlation_matrix = numeric_columns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Matrice de corrélation')
    sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='coolwarm')

if page == "Le jeu de données":
    st.subheader("Modélisation des données:")
    image = r"C:/Users/azizb/Desktop/Datatsets projet/UML.PNG"
    st.image(image, caption='Your Image Caption', use_column_width=True)
    st.subheader("Problématiques rencontrées:")
    st.write("""
       - Redondance au niveau des données.
       - Taille des fichiers.
       - Taille de la mémoire (16GO).
       - Types de données.
       """)

st.write("""
    Pour adapter les données à notre projet, nous avons effectué les étapes suivantes :
    - Nettoyage des données en supprimant les entrées avec des valeurs manquantes ou incorrectes.
    - Traitement des données redondantes pour éviter toute redondance et améliorer l'efficacité de l'analyse.
    - Normalisation des données numériques pour les mettre à l'échelle et les rendre comparables.
    - Encodage des données catégoriques pour les convertir en un format approprié pour les algorithmes de machine learning.
    - Sélection des fonctionnalités les plus pertinentes pour réduire la dimensionnalité et améliorer les performances des modèles.
    - Rééquilibrage des classes dans le cas de déséquilibres pour éviter tout biais dans l'apprentissage des modèles.
""")
              
