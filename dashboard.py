import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Classification des Ménages Haïtiens",
    page_icon="⚡",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: grey;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .high-consumption {
        border-left-color: #ff4b4b;
    }
    .medium-consumption {
        border-left-color: #ffa500;
    }
    .low-consumption {
        border-left-color: #00cc96;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class ConsumptionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = [
            'avg_amperage_per_day', 
            'avg_depense_per_day', 
            'nombre_personnes', 
            'jours_observed', 
            'ratio_depense_amperage'
        ]
    
    def load_artifacts(self, model_path, scaler_path, encoder_path):
        """Charger le modèle et les préprocesseurs"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des artefacts: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Prétraiter les données d'entrée"""
        try:
            # Créer un DataFrame avec les features attendues
            input_df = pd.DataFrame([input_data])
            
            # S'assurer que toutes les colonnes sont présentes
            for feature in self.features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Réorganiser les colonnes dans l'ordre attendu
            input_df = input_df[self.features]
            
            # Standardiser les données
            input_scaled = self.scaler.transform(input_df)
            
            return input_scaled
        except Exception as e:
            st.error(f"Erreur lors du prétraitement: {e}")
            return None
    
    def predict(self, input_data):
        """Faire une prédiction"""
        try:
            input_scaled = self.preprocess_input(input_data)
            if input_scaled is None:
                return None
            
            # Prédiction
            prediction_encoded = self.model.predict(input_scaled)[0]
            probabilities = self.model.predict_proba(input_scaled)[0]
            
            # Décoder la prédiction
            prediction_decoded = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return {
                'prediction': prediction_decoded,
                'probabilities': probabilities,
                'classes': self.label_encoder.classes_
            }
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
            return None

def main():
    # En-tête de l'application
    st.markdown('<h1 class="main-header">⚡ Classification des Ménages Haïtiens</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette application utilise un modèle de machine learning pour classifier les ménages haïtiens 
    selon leur niveau de consommation énergétique (faible, moyen, élevé).
    """)
    
    # Initialisation du prédicteur
    predictor = ConsumptionPredictor()
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choisir le mode",
        ["🔮 Prédiction Unique", "📊 Batch Prediction", "📈 Analytics", "ℹ️ A propos"]
    )
    
    # Chargement des artefacts (à adapter selon votre chemin)
    with st.sidebar.expander("Configuration du Modèle"):
        st.info("""
        Le modèle chargé est XGBoost optimisé avec:
        - F1-Score: 99.8%
        - Balanced Accuracy: 99.8%
        """)
    
    # Chemin vers vos artefacts (à modifier selon votre structure)
    model_path = "Model/best_model_20251025_2039.joblib"
    scaler_path = "Model/scaler.joblib"
    encoder_path = "Model/label_encoder.joblib"
    
    # Charger les artefacts
    if not predictor.load_artifacts(model_path, scaler_path, encoder_path):
        st.error("Impossible de charger le modèle. Vérifiez les chemins des fichiers.")
        return
    
    if app_mode == "🔮 Prédiction Unique":
        show_single_prediction(predictor)
    elif app_mode == "📊 Batch Prediction":
        show_batch_prediction(predictor)
    elif app_mode == "📈 Analytics":
        show_analytics()
    else:
        show_about()

def show_single_prediction(predictor):
    """Interface pour la prédiction unique"""
    
    st.header("🔮 Prédiction de Consommation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres du Ménage")
        
        # Formulaire de saisie
        avg_amperage = st.number_input(
            "Ampérage moyen quotidien (A)",
            min_value=0.0,
            max_value=100.0,
            value=1.5,
            step=0.1,
            help="Consommation électrique moyenne par jour"
        )
        
        avg_depense = st.number_input(
            "Dépenses moyennes quotidiennes ($)",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.01,
            help="Dépenses moyennes en électricité par jour"
        )
        
        nombre_personnes = st.number_input(
            "Nombre de personnes dans le foyer",
            min_value=1,
            max_value=20,
            value=4,
            step=1
        )
        
        jours_observed = st.number_input(
            "Nombre de jours d'observation",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Nombre de jours sur lesquels les données sont collectées"
        )
    
    with col2:
        st.subheader("Informations Complémentaires")
        
        zone = st.selectbox(
            "Zone géographique",
            ["Zone Inconnue", "Môle Saint-Nicolas", "Jean Rabel", "Bombardopolis", "Mare-Rouge"]
        )
        
        type_maison = st.selectbox(
            "Type de maison",
            ["Rezidansyel", "Apartment", "Kay modèn", "Kay tradisyonèl"]
        )
        
        # Calcul automatique du ratio
        if avg_amperage > 0:
            ratio = avg_depense / avg_amperage
        else:
            ratio = 0
        
        st.metric("Ratio Dépenses/Ampérage", f"{ratio:.4f}")
        
        # Bouton de prédiction
        if st.button("🔍 Prédire le Niveau de Consommation", type="primary"):
            # Préparation des données d'entrée
            input_data = {
                'avg_amperage_per_day': avg_amperage,
                'avg_depense_per_day': avg_depense,
                'nombre_personnes': nombre_personnes,
                'jours_observed': jours_observed,
                'ratio_depense_amperage': ratio
            }
            
            # Prédiction
            result = predictor.predict(input_data)
            
            if result:
                display_prediction_result(result, input_data)

def display_prediction_result(result, input_data):
    """Afficher les résultats de la prédiction"""
    
    prediction = result['prediction']
    probabilities = result['probabilities']
    classes = result['classes']
    
    # Déterminer la classe CSS
    if prediction == 'grand':
        css_class = "high-consumption"
        color = "#ff4b4b"
        emoji = "🔴"
    elif prediction == 'moyen':
        css_class = "medium-consumption"
        color = "#ffa500"
        emoji = "🟡"
    else:
        css_class = "low-consumption"
        color = "#00cc96"
        emoji = "🟢"
    
    # Carte de résultat
    st.markdown(f"""
    <div class="prediction-card {css_class}">
        <h2>{emoji} Prédiction: {prediction.upper()}</h2>
        <p>Le ménage est classifié comme <strong>{prediction} consommateur</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques et visualisations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Graphique en radar des probabilités
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=probabilities,
            theta=[c.capitalize() for c in classes],
            fill='toself',
            fillcolor=color,
            opacity=0.6,
            line=dict(color=color)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Probabilités de Classification",
            height=300
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Bar chart des probabilités
        fig_bar = px.bar(
            x=[c.capitalize() for c in classes],
            y=probabilities,
            color=probabilities,
            color_continuous_scale=['green', 'orange', 'red'],
            labels={'x': 'Classe', 'y': 'Probabilité'},
            title="Distribution des Probabilités"
        )
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col3:
        # Métriques détaillées
        st.metric("Confiance Maximale", f"{max(probabilities)*100:.1f}%")
        st.metric("Ampérage Quotidien", f"{input_data['avg_amperage_per_day']} A")
        st.metric("Dépenses Quotidiennes", f"${input_data['avg_depense_per_day']:.2f}")
    
    # Recommandations basées sur la prédiction
    st.subheader("🎯 Recommandations")
    
    recommendations = {
        'petit': [
            "✅ Profil de consommation efficace",
            "💡 Maintenir les bonnes habitudes de consommation",
            "📊 Surveillance standard mensuelle suffisante"
        ],
        'moyen': [
            "⚠️ Consommation dans la moyenne",
            "🔍 Analyser les opportunités d'optimisation",
            "📈 Surveiller les pics de consommation"
        ],
        'grand': [
            "🚨 Forte consommation détectée",
            "💡 Audit énergétique recommandé",
            "🔧 Optimisation des équipements énergivores",
            "📋 Plan de réduction de consommation"
        ]
    }
    
    for rec in recommendations.get(prediction, []):
        st.write(rec)

def show_batch_prediction(predictor):
    """Interface pour les prédictions par lot"""
    
    st.header("📊 Prédiction par Lot")
    
    st.info("""
    Téléchargez un fichier CSV contenant les données des ménages. 
    Le fichier doit contenir les colonnes suivantes:
    - avg_amperage_per_day
    - avg_depense_per_day  
    - nombre_personnes
    - jours_observed
    - ratio_depense_amperage (optionnel, calculé automatiquement si absent)
    """)
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            df = pd.read_csv(uploaded_file)
            st.success(f"Fichier chargé avec succès: {len(df)} enregistrements")
            
            # Aperçu des données
            st.subheader("Aperçu des Données")
            st.dataframe(df.head())
            
            # Vérification des colonnes requises
            required_columns = ['avg_amperage_per_day', 'avg_depense_per_day', 
                              'nombre_personnes', 'jours_observed']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Colonnes manquantes: {missing_columns}")
            else:
                # Calcul du ratio si absent
                if 'ratio_depense_amperage' not in df.columns:
                    df['ratio_depense_amperage'] = df['avg_depense_per_day'] / df['avg_amperage_per_day']
                    df['ratio_depense_amperage'] = df['ratio_depense_amperage'].replace([np.inf, -np.inf], 0)
                
                if st.button("🚀 Lancer les Prédictions", type="primary"):
                    with st.spinner("Traitement en cours..."):
                        predictions = []
                        probabilities_list = []
                        
                        for _, row in df.iterrows():
                            input_data = {
                                'avg_amperage_per_day': row['avg_amperage_per_day'],
                                'avg_depense_per_day': row['avg_depense_per_day'],
                                'nombre_personnes': row['nombre_personnes'],
                                'jours_observed': row['jours_observed'],
                                'ratio_depense_amperage': row['ratio_depense_amperage']
                            }
                            
                            result = predictor.predict(input_data)
                            if result:
                                predictions.append(result['prediction'])
                                probabilities_list.append(result['probabilities'])
                            else:
                                predictions.append('Erreur')
                                probabilities_list.append([0, 0, 0])
                        
                        # Ajout des résultats au DataFrame
                        df_result = df.copy()
                        df_result['niveau_conso_pred'] = predictions
                        df_result['prob_petit'] = [p[0] for p in probabilities_list]
                        df_result['prob_moyen'] = [p[1] for p in probabilities_list]
                        df_result['prob_grand'] = [p[2] for p in probabilities_list]
                        df_result['confiance'] = [max(p) for p in probabilities_list]
                        
                        # Affichage des résultats
                        st.subheader("Résultats des Prédictions")
                        st.dataframe(df_result)
                        
                        # Statistiques
                        st.subheader("📈 Statistiques des Prédictions")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            count_petit = (df_result['niveau_conso_pred'] == 'petit').sum()
                            st.metric("Petits Consommateurs", count_petit)
                        
                        with col2:
                            count_moyen = (df_result['niveau_conso_pred'] == 'moyen').sum()
                            st.metric("Moyens Consommateurs", count_moyen)
                        
                        with col3:
                            count_grand = (df_result['niveau_conso_pred'] == 'grand').sum()
                            st.metric("Grands Consommateurs", count_grand)
                        
                        with col4:
                            avg_confidence = df_result['confiance'].mean()
                            st.metric("Confiance Moyenne", f"{avg_confidence*100:.1f}%")
                        
                        # Téléchargement des résultats
                        csv = df_result.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les Résultats (CSV)",
                            data=csv,
                            file_name=f"predictions_consommation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {e}")

def show_analytics():
    """Page d'analytics et de visualisations"""
    
    st.header("📈 Analytics et Insights")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Performance du Modèle", "99.8%")
    
    with col2:
        st.metric("Précision", "99.8%")
    
    with col3:
        st.metric("Taux d'Erreur", "0.2%")
    
    with col4:
        st.metric("Données d'Entraînement", "2,716 foyers")
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des classes (exemple)
        distribution_data = {
            'Classe': ['Petit', 'Moyen', 'Grand'],
            'Pourcentage': [34.0, 33.0, 33.0]
        }
        
        fig_dist = px.pie(
            distribution_data, 
            values='Pourcentage', 
            names='Classe',
            title="Distribution des Classes de Consommation",
            color='Classe',
            color_discrete_map={'Petit': 'green', 'Moyen': 'orange', 'Grand': 'red'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Importance des features (exemple)
        importance_data = {
            'Feature': ['Ampérage Moyen', 'Dépenses Moyennes', 'Ratio', 'Jours Obs.', 'Nb Personnes'],
            'Importance': [60.5, 35.1, 3.4, 0.8, 0.2]
        }
        
        fig_imp = px.bar(
            importance_data,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Importance des Caractéristiques",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

def show_about():
    """Page À propos"""
    
    st.header("ℹ️ À Propos")
    
    st.markdown("""
    ## Classification des Ménages Haïtiens par Niveau de Consommation Énergétique
    
    ### 📋 Description du Projet
    Cette application utilise un modèle de machine learning avancé pour classifier automatiquement 
    les ménages haïtiens selon leur niveau de consommation énergétique.
    
    ### 🎯 Objectifs
    - **Segmenter** les ménages en trois catégories: petit, moyen, grand consommateur
    - **Optimiser** la planification énergétique nationale
    - **Personnaliser** les stratégies tarifaires et d'efficacité énergétique
    
    ### 🔧 Technologies Utilisées
    - **Machine Learning**: XGBoost, Random Forest, Logistic Regression
    - **Traitement des Données**: Pandas, NumPy, Scikit-learn
    - **Visualisation**: Plotly, Matplotlib
    - **Interface**: Streamlit
    - **Données**: Compteurs intelligents Sigora (Janvier 2023 - Septembre 2025)
    
    ### 📊 Performance du Modèle
    - **F1-Score**: 99.8%
    - **Balanced Accuracy**: 99.8%
    - **Précision**: 99.8%
    - **Taux d'Erreur**: 0.2%
    
    ### 👥 Équipe
    - Saint Germain Emode
    - Darlens Damisca
    
    ### 📞 Contact
    Pour toute question ou suggestion, contactez-nous:
    - ger-modeel2@gmail.com
    - bdamisca96@gmail.com
    """)

if __name__ == "__main__":
    main()
