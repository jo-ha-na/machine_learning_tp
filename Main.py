import pandas as pd 
import csv
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score
import os

from sklearn.cluster import KMeans

import seaborn as sns 
def createCSV():
    # Fichiers d'entrée et de sortie
    input_file = 'ValeursFoncieres-2023.txt'
    output_file = 'ValeursFoncieres_93000.csv'

    # Ouvrir le fichier d'entrée en lecture et le fichier CSV en écriture
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='|')  # Modifier le délimiteur si besoin
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            # Vérifier si la ligne a au moins 17 colonnes
            if len(row) >= 17:
                # Champ 17 correspond au code postal (index 16)
                zip_code = row[16]
                
                # Si le code postal est 93000, on écrit la ligne dans le fichier de sortie
                if zip_code == '93000':
                    writer.writerow(row)

    print(f"Le tri est terminé. Les lignes avec le code postal 93000 sont enregistrées dans {output_file}")
# Count the missing values in each column 
def graph_missing_values(df):
    df = pd.read_csv('93000.csv') 
    missing_values = 100 - df.isnull().sum() / len(df) * 100

    # Filter columns that have at least one missing value 


    # Plotting the missing values using seaborn for better visualization 

    plt.figure(figsize=(12, 6)) 

    sns.barplot(x=missing_values.index, y=missing_values.values) 

    plt.xticks(rotation=90) 

    plt.xlabel('Columns') 

    plt.ylabel('Percentage of Values') 

    plt.title('Values Count per Column') 

    plt.tight_layout() 

    plt.show() 

def deleteEmptyCells(df, column):
    df[column] = df[column].replace('', np.nan)
    df.dropna(subset=[column], inplace=True)
    return df

def cleanCSV():
    df = pd.read_csv('93000.csv') 
    
    colonnes_importantes = [
        'Date mutation',           # Exemple : Date de la transaction
        'Nature mutation',         # Nature de la mutation
        'Valeur fonciere',         # Valeur foncière (cible)
        'Code postal',             # Code postal
        'Code departement',        # Département
        'Type de voie',               # Type de voie
        'Voie',                    # Voie
        'Surface reelle bati',    # Surface réelle bâtie
        'Nombre pieces principales', # Nombre de pièces principales
        'Type local',
        'Code type local'         # Type de local (maison, appartement, etc.)
        'Surface terrain'          # Surface du terrain
    ]

    df = df[colonnes_importantes]

    df = deleteEmptyCells(df, "Valeur fonciere")
    df = deleteEmptyCells(df, "Date mutation")
    df = deleteEmptyCells(df, "Type local")
    df = deleteEmptyCells(df, "Surface reelle bati")
    df = deleteEmptyCells(df, "Nombre pieces principales")
    df = deleteEmptyCells(df, "Surface terrain")

    df["Type local"] = df["Type local"].replace('', 'Inconnue')
    df["Type de voie"] = df["Type de voie"].replace('', 'Inconnue')
    df["Nature mutation"] = df["Nature mutation"].replace('', 'Inconnue')

    df.to_csv('clean_93000.csv', index=False)  

def remove_outliers(df, z_threshold=4):
    # Supprimer les outliers basés sur un seuil z
    print("Suppression des outliers...")
    from scipy import stats
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df))
    df_no_outliers = df[(z_scores < z_threshold).all(axis=1)]
   
    if df_no_outliers.empty:
        print("Attention : Toutes les lignes ont été supprimées comme outliers. Réduction du seuil Z pour conserver plus de données.")
        return df
   
    print(f"Outliers supprimés. Nombre de lignes restantes : {len(df_no_outliers)}")
    return df_no_outliers





def preparation():
    df = pd.read_csv('clean_93000.csv') 
    df = remove_outliers(df)

    df = pd.get_dummies(df, columns=['Type de voie'], prefix='voie')
    df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.')

    df['Surface terrain'] = df['Surface terrain'].astype(str)
    df['Surface terrain'] = df['Surface terrain'].str.replace(',', '.')
    df['Surface terrain'] = pd.to_numeric(df['Surface terrain'], errors='coerce')

    df['Surface reelle bati'] = df['Surface reelle bati'].astype(str)
    df['Surface reelle bati'] = df['Surface reelle bati'].str.replace(',', '.')
    df['Surface reelle bati'] = pd.to_numeric(df['Surface reelle bati'], errors='coerce')


    
    df.to_csv('clean_93000_prepared.csv', index=False)  



def find_optimal_clusters(df, max_clusters=10):
    # Normaliser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Liste pour stocker l'inertie pour chaque nombre de clusters
    inertias = []

    # Essayer différents nombres de clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)

    # Tracer la courbe d'inertie pour visualiser le "coude"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.xlabel('Nombre de Clusters (k)')
    plt.ylabel('Inertie (SSE)')
    plt.title('Méthode du Coude pour Trouver le Nombre Optimal de Clusters')
    plt.grid()
    plt.show()

def segment_data_kmeans(df, n_clusters=5):
    # Appliquer K-means pour segmenter les données
    print("Segmentation des données avec K-means...")

    # Normaliser les données avant de les passer à K-means
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    print("Segmentation par K-means réussie.")
    return df

def boxplots(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Valeur fonciere', data=df)

    # Appliquer une échelle logarithmique sur l'axe Y
    plt.yscale('log')

    plt.xlabel('Cluster')
    plt.ylabel('Valeur Foncière (échelle logarithmique)')
    plt.title('Boxplot des Valeurs Foncières par Cluster avec Échelle Logarithmique')
    plt.show()

def regression(data):
    # Feature engineering: selection des attributs importants
    features = ['Surface reelle bati', 'Surface terrain']
    X = data[features].values
    y = data['Valeur fonciere'].values
    
    # Split data 70%-30% into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Train and evaluate selected models
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        predictions = model.predict(X_test)
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        # Print evaluation metrics
        print(f"{name}:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}\n")
        # Plot predictions vs actual labels
        plt.scatter(y_test, predictions)
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{name} - Prédictions des Valeurs Foncières')
        # overlay the regression line
        z = np.polyfit(y_test, predictions, 1)
        p = np.poly1d(z)
        plt.yscale('log')
        plt.xscale('log')

        plt.plot(y_test, p(y_test), color='magenta')
        plt.show()
    
    # Visualize the Decision Tree model
    tree = export_text(models["Decision Tree Regressor"], feature_names=features)
    print(tree)



def main():
    
    #createCSV()
    #cleanCSV()
    #preparation()
    df = pd.read_csv('clean_93000_prepared.csv')
    #find_optimal_clusters(df[["Valeur fonciere", "Surface reelle bati", 'Surface terrain']])
    segment_data_kmeans(df[["Valeur fonciere", "Surface reelle bati", 'Surface terrain']])
    #boxplots(df)
    regression(df)
main()


