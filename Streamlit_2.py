# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:31:57 2024

@author: 33645
"""

import streamlit as st
from surprise import SVD, Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Charger les données
df = pd.read_csv('df.csv')
svd_df = pd.read_csv('svd.csv')
model = load_model('mod.h5')

# Titre principal
st.title("Projet de classification de films")

# Sidebar pour la navigation
st.sidebar.title("Sommaire")
pages = ["Le projet", "Le jeu de données", "Quelques visualisations", 
         "Préparation des données", "Modélisation", "Machine Learning", 
         "Conclusion et perspectives"]
page = st.sidebar.radio("Aller vers", pages)

# Page de modélisation
if page == "Modélisation":
    st.header("Modélisation")
    
    # Modèle KNN
    st.subheader("Modèle KNN")
    st.write("""
    Le modèle KNN (K-Nearest Neighbors) est un algorithme de recommandation basé sur la similarité. 
    Il recommande des films en fonction des notes de films similaires.
    """)
    st.write("**Avantages**")
    st.write("""
    - Simple à comprendre et à implémenter.
    - Ne nécessite pas d'entraînement complexe.
    """)
    st.write("**Inconvénients**")
    st.write("""
    - Peut être lent pour des grands datasets.
    - Performance dépend fortement du choix de K et de la métrique de similarité.
    """)

    # Exemple de code pour KNN
    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(svd_df[['userId', 'movieId', 'note']], reader)
    knn = KNNBasic()
    knn_results = cross_validate(knn, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    st.write("**Résultats KNN (Cross-Validation)**")
    st.write(pd.DataFrame(knn_results).mean(axis=0))

    # Modèle SVD
    st.subheader("Modèle SVD")
    st.write("""
    Le modèle SVD (Singular Value Decomposition) est un algorithme de filtrage collaboratif basé sur la décomposition matricielle.
    Il factorise la matrice des notes pour découvrir les relations latentes entre les utilisateurs et les films.
    """)
    st.write("**Avantages**")
    st.write("""
    - Peut capturer des relations latentes complexes.
    - Souvent performant sur les systèmes de recommandation.
    """)
    st.write("**Inconvénients**")
    st.write("""
    - Peut être complexe à implémenter et à comprendre.
    - Nécessite un temps d'entraînement plus long.
    """)
    
    # Exemple de code pour SVD
    svd = SVD()
    svd_results = cross_validate(svd, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    st.write("**Résultats SVD (Cross-Validation)**")
    st.write(pd.DataFrame(svd_results).mean(axis=0))

# Page de machine learning
if page == "Machine Learning":
    st.header("Machine Learning")

    # Charger le DataFrame contenant les noms des films
    movie_names_df = pd.read_csv('movies.csv')  # Supposons que ce fichier contient les colonnes 'movieId' et 'titre'

    # Vérifiez les noms des colonnes
    st.write("Colonnes de movie_names_df:", movie_names_df.columns)

    # Nettoyer les valeurs dans la colonne 'movieId' de movie_names_df
    movie_names_df['movieId'] = movie_names_df['movieId'].str.extract('(\d+)').astype(int)

    # Assurez-vous que les colonnes 'movieId' des deux DataFrames sont du même type
    df['movieId'] = df['movieId'].astype(int)

    # Créer un dictionnaire de mapping de movieId à titre
    movie_id_to_title = pd.Series(movie_names_df['titre'].values, index=movie_names_df['movieId']).to_dict()

    # Préparation des données comme dans votre modèle
    reader = Reader(rating_scale=(0, 9))
    data = Dataset.load_from_df(svd_df[['userId', 'movieId', 'note']], reader)
    svd = SVD()

    trainset = data.build_full_trainset()
    svd.fit(trainset)

    user_latent_matrix = svd.pu
    movie_latent_matrix = svd.qi

    user_id_to_index = {uid: idx for idx, uid in enumerate(svd.trainset._raw2inner_id_users)}
    movie_id_to_index = {mid: idx for idx, mid in enumerate(svd.trainset._raw2inner_id_items)}

    X = df.drop(['note'], axis=1)

    def get_combined_features(userId, movieId):
        if userId in user_id_to_index and movieId in movie_id_to_index:
            user_features = user_latent_matrix[user_id_to_index[userId]]
            movie_features = movie_latent_matrix[movie_id_to_index[movieId]]
            other_features = X.loc[(X['userId'] == userId) & (X['movieId'] == movieId)].drop(['userId', 'movieId'], axis=1).values.flatten()

            if other_features.size > 0:
                all_features = np.concatenate([user_features, movie_features, other_features])
                return all_features
        return None

    def get_top_n_recommendations(userId, n=5):
        if userId in user_id_to_index:
            user_features = user_latent_matrix[user_id_to_index[userId]]
            scores = np.dot(movie_latent_matrix, user_features)
            movie_indices = np.argsort(scores)[-n:][::-1]
            top_movies = svd.trainset._raw2inner_id_items.keys()
            top_movie_ids = [list(top_movies)[i] for i in movie_indices]
            top_movies_df = df[df['movieId'].isin(top_movie_ids)]
            
            # Faire une jointure pour inclure les noms des films
            top_movies_df = top_movies_df.merge(movie_names_df, on='movieId', how='left')
            
            # Ajouter les prédictions
            top_movies_df['prediction'] = top_movies_df['movieId'].apply(lambda x: scores[movie_id_to_index[x]])
            
            # Supprimer les doublons
            top_movies_df = top_movies_df.drop_duplicates(subset=['movieId'])
            
            return top_movies_df
        return pd.DataFrame()

    # Titre de l'application
    st.subheader('Démonstration des prédictions du modèle')

    # Sélection des utilisateurs et des films
    st.write("Sélectionnez un utilisateur et un film pour voir la prédiction:")

    user_id = st.selectbox("User ID", sorted(svd_df['userId'].unique()))

    # Créer une liste des options pour le sélecteur de films, affichant le titre du film
    movie_options = {movie_id: f"{movie_id} - {title}" for movie_id, title in movie_id_to_title.items() if movie_id in svd_df['movieId'].unique()}
    movie_id = st.selectbox("Movie ID", options=sorted(movie_options.keys()), format_func=lambda x: movie_options[x])

    # Bouton de prédiction
    if st.button("Prédire"):
        features = get_combined_features(user_id, movie_id)
        
        if features is not None:
            # Normalisation des caractéristiques
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features.reshape(1, -1))
            
            # Prédiction du modèle
            prediction = model.predict(features_scaled)
            predicted_class = np.argmax(prediction, axis=1)
            
            st.write(f"Prédiction pour l'utilisateur {user_id} et le film {movie_id_to_title[movie_id]} : Classe {predicted_class[0]}")
        else:
            st.write("Erreur lors de la récupération des caractéristiques pour cet utilisateur et ce film.")

    # Recommandations de films
    st.write("Recommandations de films pour l'utilisateur sélectionné:")

    recommendations_df = get_top_n_recommendations(user_id)

    if not recommendations_df.empty:
        st.write(recommendations_df[['movieId', 'titre', 'tag', 'Directors', 'Principal', 'runtimeMinutes', 'prediction']].head(5))
    else:
        st.write("Aucune recommandation disponible pour cet utilisateur.")