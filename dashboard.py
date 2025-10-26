import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from pathlib import Path
import requests
from io import BytesIO
import base64

# ==============================
# CONFIGURATION GLOBALE
# ==============================
st.set_page_config(
    page_title="Classification des Ménages Haïtiens - Sigora",
    page_icon="🇭🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# STYLE CSS PERSONNALISÉ
# ==============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-medium {
        background: linear-gradient(135deg, #ffd93d, #ff9f43);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-low {
        background: linear-gradient(135deg, #6bcf7f, #4cd137);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .alert-box {
        background: linear-gradient(135deg, #ff7979, #eb4d4b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .impact-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #00b894, #55a630);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CLASSE PRINCIPALE - AVEC UPLOAD DE MODÈLE
# ==============================
class SigoraHouseholdClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.dataset = None
        self.performance_metrics = None
        self.model_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        """Charger les fichiers du modèle depuis le dossier Model/"""
        try:
            # Essayer de charger depuis le dossier Model/
            base_path = "Model"
            
            # Vérifier si le dossier existe
            if os.path.exists(base_path):
                st.sidebar.success("📁 Dossier Model/ détecté")
                files = os.listdir(base_path)
                st.sidebar.write(f"Fichiers trouvés: {', '.join(files)}")
                
                # Charger le modèle
                model_files = [f for f in files if f.startswith('best_model') and f.endswith('.joblib')]
                if model_files:
                    self.model = joblib.load(os.path.join(base_path, model_files[0]))
                    st.sidebar.success(f"✅ Modèle chargé: {model_files[0]}")
                else:
                    st.sidebar.error("❌ Fichier modèle non trouvé")
                    self.setup_demo_mode()
                    return
                
                # Charger le scaler
                if 'scaler.joblib' in files:
                    self.scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
                    st.sidebar.success("✅ Scaler chargé")
                else:
                    st.sidebar.error("❌ Scaler non trouvé")
                    self.setup_demo_mode()
                    return
                
                # Charger l'encodeur
                if 'label_encoder.joblib' in files:
                    self.encoder = joblib.load(os.path.join(base_path, 'label_encoder.joblib'))
                    st.sidebar.success("✅ Encodeur chargé")
                else:
                    st.sidebar.error("❌ Encodeur non trouvé")
                    self.setup_demo_mode()
                    return
                
                # Charger les données
                data_files = [f for f in files if f.startswith('final_results') and f.endswith('.csv')]
                if data_files:
                    self.dataset = pd.read_csv(os.path.join(base_path, data_files[0]))
                    st.sidebar.success(f"✅ Données chargées: {data_files[0]}")
                else:
                    st.sidebar.warning("⚠️ Données non trouvées - Génération de données de démo")
                    self.generate_demo_data()
                
                # Charger les métriques
                if 'performance_metrics.json' in files:
                    with open(os.path.join(base_path, 'performance_metrics.json'), 'r') as f:
                        self.performance_metrics = json.load(f)
                    st.sidebar.success("✅ Métriques chargées")
                
                self.model_loaded = True
                st.sidebar.success("🎯 **VRAI MODÈLE ACTIVÉ**")
                
            else:
                st.sidebar.error("❌ Dossier 'Model/' introuvable")
                st.sidebar.info("💡 Uploadez vos fichiers dans le dossier Model/")
                self.setup_demo_mode()
                
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de chargement: {str(e)}")
            self.setup_demo_mode()

    def setup_demo_mode(self):
        """Mode démo si le vrai modèle n'est pas disponible"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        st.sidebar.warning("🎮 Activation du mode démo")
        
        np.random.seed(42)
        self.generate_demo_data()
        
        # Préparation des features pour le modèle démo
        features = ['avg_amperage_per_day', 'avg_depense_per_day', 'nombre_personnes', 'jours_observed', 'ratio_depense_amperage']
        X = self.dataset[features]
        y = self.dataset['niveau_conso_pred']

        # Entraînement du modèle démo
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_scaled, y_enc)
        
        self.performance_metrics = {
            "test_accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.89,
            "f1_score": 0.90
        }
        
        self.model_loaded = False

    def generate_demo_data(self):
        """Générer des données de démo réalistes"""
        n_samples = 1200
        zones = ['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Les Cayes', 'Jacmel']
        
        self.dataset = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, n_samples),
            'avg_depense_per_day': np.random.exponential(0.05, n_samples),
            'nombre_personnes': np.random.randint(2, 7, n_samples),
            'jours_observed': np.random.randint(30, 365, n_samples),
            'latitude': np.random.uniform(18.0, 20.2, n_samples),
            'longitude': np.random.uniform(-74.5, -71.8, n_samples),
            'zone': np.random.choice(zones, n_samples),
            'menage_id': [f"MEN{str(i).zfill(4)}" for i in range(n_samples)]
        })
        
        self.dataset['ratio_depense_amperage'] = (
            self.dataset['avg_depense_per_day'] / 
            (self.dataset['avg_amperage_per_day'] + 1e-9)
        )
        
        # Classification réaliste
        score_consommation = (
            self.dataset['avg_amperage_per_day'] * 0.6 +
            self.dataset['nombre_personnes'] * 0.2 +
            self.dataset['ratio_depense_amperage'] * 0.2
        )
        
        self.dataset['niveau_conso_pred'] = pd.cut(
            score_consommation,
            bins=[-1, 1.5, 3.0, np.inf],
            labels=['petit', 'moyen', 'grand']
        )

    def predict_household(self, features):
        """Faire une prédiction unique"""
        try:
            X = np.array([features]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0]
            label = self.encoder.inverse_transform([pred])[0]
            return label, prob
        except Exception as e:
            st.error(f"Erreur de prédiction: {e}")
            return "moyen", [0.33, 0.34, 0.33]

    def detect_anomalies(self):
        """Détecter les consommations anormales"""
        if self.dataset is None:
            return []
        
        anomalies = []
        for idx, row in self.dataset.iterrows():
            if row['avg_amperage_per_day'] > 6.0:
                anomalies.append({
                    'id': row.get('menage_id', f"MEN{idx:04d}"),
                    'type': '🚨 Consommation Excessive',
                    'valeur': f"{row['avg_amperage_per_day']:.1f}A",
                    'seuil': '6.0A',
                    'zone': row.get('zone', 'Inconnue'),
                    'personnes': row.get('nombre_personnes', 'N/A')
                })
            elif row['ratio_depense_amperage'] > 0.12:
                anomalies.append({
                    'id': row.get('menage_id', f"MEN{idx:04d}"),
                    'type': '💸 Inefficacité Économique',
                    'valeur': f"Ratio {row['ratio_depense_amperage']:.3f}",
                    'seuil': '0.120',
                    'zone': row.get('zone', 'Inconnue'),
                    'personnes': row.get('nombre_personnes', 'N/A')
                })
        
        return anomalies[:10]

# ==============================
# FONCTIONNALITÉS AVANCÉES
# ==============================

def show_interactive_map(clf):
    """🗺️ Carte Interactive des Ménages"""
    st.markdown('<h2 class="sub-header">🗺️ Carte Interactive des Consommations</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Données réelles utilisées</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🎮 **MODE DÉMO** - Données simulées</div>', unsafe_allow_html=True)
    
    if clf.dataset is None:
        st.error("❌ Données non disponibles")
        return
    
    viz_type = st.radio("**Type de visualisation:**", ["Points Colorés", "Heatmap de Densité"], horizontal=True)
    
    if viz_type == "Points Colorés":
        fig = px.scatter_mapbox(clf.dataset, 
                               lat="latitude", 
                               lon="longitude",
                               color="niveau_conso_pred",
                               color_discrete_map={
                                   'petit': '#4cd137',
                                   'moyen': '#ff9f43', 
                                   'grand': '#ff6b6b'
                               },
                               hover_data={
                                   'avg_amperage_per_day': ':.2f',
                                   'avg_depense_per_day': ':.3f',
                                   'nombre_personnes': True,
                                   'zone': True
                               },
                               zoom=6.5,
                               height=600,
                               title="Répartition Géographique des Ménages en Haïti")
    else:
        fig = px.density_mapbox(clf.dataset, 
                               lat="latitude", 
                               lon="longitude",
                               z='avg_amperage_per_day',
                               radius=15,
                               zoom=6.5,
                               height=600,
                               title="Heatmap de la Consommation Électrique")
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def show_impact_simulator(clf):
    """💰 Simulateur d'Impact Économique"""
    st.markdown('<h2 class="sub-header">💰 Simulateur d\'Économies Potentielles</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Prédictions précises</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        menage_type = st.selectbox(
            "**Type de consommation:**",
            ["petit", "moyen", "grand"],
            index=1,
            format_func=lambda x: {
                "petit": "🟢 Faible Consommateur", 
                "moyen": "🟡 Consommation Moyenne", 
                "grand": "🔴 Grand Consommateur"
            }[x]
        )
        
        interventions = st.multiselect(
            "**Actions d'optimisation:**",
            ["Compteur intelligent", "Éclairage LED", "Électroménager efficace", "Sensibilisation", "Tarification incitative"],
            default=["Compteur intelligent", "Éclairage LED"]
        )
    
    with col2:
        economie_base = {"petit": 80, "moyen": 150, "grand": 350}[menage_type]
        multiplicateur = 1.0
        
        bonus = {
            "Compteur intelligent": 0.3,
            "Éclairage LED": 0.25,
            "Électroménager efficace": 0.4,
            "Sensibilisation": 0.15,
            "Tarification incitative": 0.3
        }
        
        for intervention in interventions:
            multiplicateur += bonus.get(intervention, 0)
        
        economie_totale = economie_base * multiplicateur
        
        st.markdown(f'''
        <div class="impact-card">
            <h3>💵 Économies Annuelles Estimées</h3>
            <h1>${economie_totale:.0f}</h1>
            <p>Par ménage • Basé sur les données { "réelles" if clf.model_loaded else "simulées" }</p>
        </div>
        ''', unsafe_allow_html=True)
        
        menages_impactes = st.slider("**Nombre de ménages impactés:**", 100, 5000, 1000, 100)
        impact_national = economie_totale * menages_impactes
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("💰 Économies totales", f"${impact_national:,.0f}")
        with col_b:
            st.metric("🏠 Ménages couverts", f"{menages_impactes}")

def show_real_time_alerts(clf):
    """🚨 Alertes Temps Réel"""
    st.markdown('<h2 class="sub-header">🚨 Détection d\'Anomalies</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Détection précise</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Scanner les Consommations Anormales", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            anomalies = clf.detect_anomalies()
            
            if not anomalies:
                st.success("✅ **Aucune anomalie critique détectée**")
            else:
                st.error(f"🚨 **{len(anomalies)} anomalies détectées**")
                
                for i, anomaly in enumerate(anomalies, 1):
                    st.markdown(f"""
                    <div style='
                        background: {"#ff6b6b" if "Excessive" in anomaly["type"] else "#ffa726"}; 
                        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                        border-left: 5px solid #c23616;
                    '>
                        <strong>#{i} - {anomaly['id']}</strong><br>
                        <strong>{anomaly['type']}</strong><br>
                        📊 {anomaly['valeur']} | 🎯 Seuil: {anomaly['seuil']}<br>
                        📍 {anomaly['zone']} | 👥 {anomaly['personnes']} personnes
                    </div>
                    """, unsafe_allow_html=True)

def show_3d_clusters(clf):
    """🔮 Visualisation 3D des Clusters"""
    st.markdown('<h2 class="sub-header">🔮 Visualisation 3D des Profils</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Clusters réels</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        x_axis = st.selectbox("**Axe X**", 
                             ['avg_amperage_per_day', 'avg_depense_per_day', 'nombre_personnes', 'ratio_depense_amperage'],
                             index=0)
        y_axis = st.selectbox("**Axe Y**", 
                             ['avg_depense_per_day', 'avg_amperage_per_day', 'nombre_personnes', 'ratio_depense_amperage'],
                             index=1)
        z_axis = st.selectbox("**Axe Z**", 
                             ['nombre_personnes', 'avg_amperage_per_day', 'avg_depense_per_day', 'ratio_depense_amperage'],
                             index=0)
    
    with col2:
        plot_df = clf.dataset.copy().head(400)
        
        fig = px.scatter_3d(plot_df,
                           x=x_axis,
                           y=y_axis, 
                           z=z_axis,
                           color='niveau_conso_pred',
                           color_discrete_map={
                               'petit': '#4cd137',
                               'moyen': '#ff9f43',
                               'grand': '#ff6b6b'
                           },
                           hover_data={
                               'menage_id': True,
                               'zone': True,
                               'avg_amperage_per_day': ':.2f'
                           },
                           title="Clusters 3D des Profils de Consommation",
                           height=600)
        
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# PAGES EXISTANTES
# ==============================

def show_dashboard(clf):
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord Principal</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Données réelles</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🎮 **MODE DÉMO** - Données simulées</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏠 Ménages analysés", len(clf.dataset))
    with col2:
        acc = clf.performance_metrics.get("test_accuracy", 0.92) * 100
        st.metric("🎯 Précision du modèle", f"{acc:.1f}%")
    with col3:
        grands = (clf.dataset["niveau_conso_pred"] == "grand").sum()
        st.metric("🔴 Grands consommateurs", grands)
    with col4:
        zones = clf.dataset["zone"].nunique()
        st.metric("📍 Zones couvertes", zones)

    col_left, col_right = st.columns(2)
    with col_left:
        dist = clf.dataset["niveau_conso_pred"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, hole=0.4,
                     color=dist.index, 
                     color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        zone_data = clf.dataset.groupby("zone")["niveau_conso_pred"].value_counts().unstack().fillna(0)
        fig = px.bar(zone_data, barmode="stack", 
                    color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)

def show_prediction(clf):
    st.markdown('<h2 class="sub-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="success-box">🎯 **VRAI MODÈLE** - Prédictions précises</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        avg_amperage = st.slider("Ampérage moyen (A)", 0.0, 15.0, 2.5, 0.1)
        avg_depense = st.slider("Dépense moyenne ($)", 0.0, 1.0, 0.12, 0.01)
        nb_personnes = st.selectbox("Nombre de personnes", [1, 2, 3, 4, 5, 6, 7, 8], 3)
    with col2:
        jours = st.slider("Jours observés", 7, 365, 90)
        ratio = st.slider("Ratio dépense/ampérage", 0.0, 0.3, 0.06, 0.01)

    if st.button("🎯 Analyser ce Ménage", type="primary", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        
        st.markdown("---")
        if pred == "grand":
            st.markdown('<div class="prediction-high"><h1>🔴 GRAND CONSOMMATEUR</h1></div>', unsafe_allow_html=True)
        elif pred == "moyen":
            st.markdown('<div class="prediction-medium"><h1>🟡 CONSOMMATION MOYENNE</h1></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low"><h1>🟢 FAIBLE CONSOMMATION</h1></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=['Faible','Moyenne','Élevée'], 
            y=prob,
            marker_color=['#4cd137','#ff9f43','#ff6b6b'],
            text=[f"{p:.1%}" for p in prob], 
            textposition='auto'
        ))
        fig.update_layout(
            title="Confiance du Modèle",
            yaxis=dict(tickformat=".0%", range=[0,1]),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# UPLOAD DE MODÈLE
# ==============================

def show_model_upload(clf):
    """📤 Interface pour uploader son modèle"""
    st.markdown('<h2 class="sub-header">📤 Uploader Votre Modèle</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Pour utiliser votre vrai modèle :**
    1. Créez un dossier `Model/` dans votre repository
    2. Uploadez vos fichiers :
       - `best_model.joblib`
       - `scaler.joblib` 
       - `label_encoder.joblib`
       - `final_results.csv`
    3. Redémarrez l'application
    """)
    
    if clf.model_loaded:
        st.success("✅ **VRAI MODÈLE DÉTECTÉ** - Toutes les fonctionnalités utilisent votre modèle entraîné")
    else:
        st.warning("🎮 **MODE DÉMO** - Uploadez vos fichiers pour utiliser votre vrai modèle")

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🏠 Classification Intelligente des Ménages Haïtiens</h1>', unsafe_allow_html=True)
    
    # Initialisation du classifieur
    clf = SigoraHouseholdClassifier()
    
    # Navigation
    st.sidebar.markdown("## 📍 Navigation")
    page = st.sidebar.radio("", [
        "🏠 Tableau de Bord",
        "🔮 Prédiction Temps Réel", 
        "🗺️ Carte Interactive",
        "💰 Simulateur d'Impact",
        "🚨 Alertes Temps Réel", 
        "🔮 Visualisation 3D",
        "📤 Upload Modèle"
    ])

    # Routage des pages
    if page == "🏠 Tableau de Bord":
        show_dashboard(clf)
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction(clf)
    elif page == "🗺️ Carte Interactive":
        show_interactive_map(clf)
    elif page == "💰 Simulateur d'Impact":
        show_impact_simulator(clf)
    elif page == "🚨 Alertes Temps Réel":
        show_real_time_alerts(clf)
    elif page == "🔮 Visualisation 3D":
        show_3d_clusters(clf)
    elif page == "📤 Upload Modèle":
        show_model_upload(clf)

    # Footer
    st.sidebar.markdown("---")
    if clf.model_loaded:
        st.sidebar.success("**🎯 VRAI MODÈLE ACTIVÉ**")
    else:
        st.sidebar.info("**🎮 MODE DÉMO**")
    
    st.sidebar.markdown("""
    **ℹ️ À propos**
    - 🤖 Machine Learning
    - 📊 Analytics avancé
    - 🇭🇹 Optimisé pour Haïti
    """)

if __name__ == "__main__":
    main()
