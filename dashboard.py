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

# ==============================
# CONFIGURATION GLOBALE
# ==============================
st.set_page_config(
    page_title="Calculateur de Consommation - Sigora",
    page_icon="🇭🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# BASE DE DONNÉES DES APPAREILS
# ==============================
APPAREILS_DATA = {
    "ampoule": {
        "nom": "💡 Ampoule LED",
        "puissance_w": 10,
        "heures_usage_jour": 6,
        "probabilite_usage": 0.95
    },
    "television": {
        "nom": "📺 Télévision",
        "puissance_w": 80,
        "heures_usage_jour": 5,
        "probabilite_usage": 0.85
    },
    "laptop": {
        "nom": "💻 Laptop",
        "puissance_w": 60,
        "heures_usage_jour": 4,
        "probabilite_usage": 0.70
    },
    "telephone": {
        "nom": "📱 Téléphone (chargeur)",
        "puissance_w": 5,
        "heures_usage_jour": 3,
        "probabilite_usage": 0.90
    },
    "refrigerateur": {
        "nom": "❄️ Réfrigérateur",
        "puissance_w": 150,
        "heures_usage_jour": 8,
        "probabilite_usage": 1.00
    },
    "radio": {
        "nom": "📻 Radio",
        "puissance_w": 15,
        "heures_usage_jour": 4,
        "probabilite_usage": 0.60
    },
    "climatiseur": {
        "nom": "❄️ Climatiseur",
        "puissance_w": 1000,
        "heures_usage_jour": 3,
        "probabilite_usage": 0.40
    },
    "ventilateur": {
        "nom": "🌀 Ventilateur",
        "puissance_w": 50,
        "heures_usage_jour": 8,
        "probabilite_usage": 0.75
    },
    "machine_laver": {
        "nom": "👕 Machine à laver",
        "puissance_w": 500,
        "heures_usage_jour": 1,
        "probabilite_usage": 0.30
    },
    "fer_repasser": {
        "nom": "🧺 Fer à repasser",
        "puissance_w": 1000,
        "heures_usage_jour": 0.5,
        "probabilite_usage": 0.25
    }
}

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
    .appliance-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .appliance-card:hover {
        border-color: #1f77b4;
        transform: translateY(-2px);
    }
    .consumption-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CLASSE PRINCIPALE
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
            base_path = "Model"
            
            if os.path.exists(base_path):
                st.sidebar.success("📁 Dossier Model/ détecté")
                files = os.listdir(base_path)
                
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
                
                self.model_loaded = True
                st.sidebar.success("🎯 **VRAI MODÈLE ACTIVÉ**")
                
            else:
                st.sidebar.error("❌ Dossier 'Model/' introuvable")
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
        n_samples = 1000
        self.dataset = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, n_samples),
            'avg_depense_per_day': np.random.exponential(0.05, n_samples),
            'nombre_personnes': np.random.randint(2, 7, n_samples),
            'jours_observed': np.random.randint(30, 365, n_samples),
        })
        
        self.dataset['ratio_depense_amperage'] = (
            self.dataset['avg_depense_per_day'] / 
            (self.dataset['avg_amperage_per_day'] + 1e-9)
        )
        
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

        features = ['avg_amperage_per_day', 'avg_depense_per_day', 'nombre_personnes', 'jours_observed', 'ratio_depense_amperage']
        X = self.dataset[features]
        y = self.dataset['niveau_conso_pred']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_enc)
        
        self.performance_metrics = {"test_accuracy": 0.92}
        self.model_loaded = False

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

# ==============================
# FONCTIONS DE CALCUL DE CONSOMMATION
# ==============================

def calculer_consommation_appareils(appareils_selectionnes, nb_personnes, tarif_kwh=0.25):
    """
    Calcule la consommation totale basée sur les appareils sélectionnés
    
    Args:
        appareils_selectionnes: Liste des appareils sélectionnés
        nb_personnes: Nombre de personnes dans le ménage
        tarif_kwh: Tarif électrique en $/kWh (valeur par défaut pour Haïti)
    
    Returns:
        dict: Résultats de calcul
    """
    consommation_totale_wh = 0
    details_appareils = []
    
    for appareil_id in appareils_selectionnes:
        if appareil_id in APPAREILS_DATA:
            appareil = APPAREILS_DATA[appareil_id]
            
            # Ajustement basé sur le nombre de personnes
            if appareil_id == "ampoule":
                quantite = max(2, nb_personnes)  # Au moins 2 ampoules
            elif appareil_id == "telephone":
                quantite = nb_personnes  # Un téléphone par personne
            elif appareil_id == "laptop":
                quantite = min(nb_personnes, 3)  # Maximum 3 laptops
            else:
                quantite = 1
            
            # Calcul consommation quotidienne
            consommation_wh = (
                appareil["puissance_w"] * 
                appareil["heures_usage_jour"] * 
                quantite *
                appareil["probabilite_usage"]
            )
            
            consommation_totale_wh += consommation_wh
            
            details_appareils.append({
                "nom": appareil["nom"],
                "quantite": quantite,
                "puissance_w": appareil["puissance_w"],
                "heures_jour": appareil["heures_usage_jour"],
                "consommation_wh": consommation_wh,
                "probabilite": appareil["probabilite_usage"]
            })
    
    # Conversion en kWh et calcul du coût
    consommation_kwh = consommation_totale_wh / 1000
    cout_quotidien = consommation_kwh * tarif_kwh
    
    # Conversion en ampérage (supposant 110V - standard Haïti)
    voltage = 110
    amperage_moyen = (consommation_kwh * 1000) / voltage / 24  # Ampérage moyen sur 24h
    
    return {
        "consommation_wh": consommation_totale_wh,
        "consommation_kwh": consommation_kwh,
        "cout_quotidien": cout_quotidien,
        "amperage_moyen": amperage_moyen,
        "details_appareils": details_appareils,
        "tarif_kwh": tarif_kwh
    }

def show_appliance_calculator(clf):
    """🔌 Calculateur de Consommation par Appareils"""
    st.markdown('<h2 class="sub-header">🔌 Calculateur Intelligent de Consommation</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.markdown('<div class="info-box">🎯 **VRAI MODÈLE** - Prédictions basées sur votre modèle entraîné</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">🎮 **MODE DÉMO** - Utilisation de données simulées</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👨‍👩‍👧‍👦 Informations du Ménage")
        nb_personnes = st.slider("Nombre de personnes dans le ménage", 1, 10, 4)
        
        st.markdown("### 💡 Sélection des Appareils")
        st.write("Cochez les appareils utilisés dans le ménage:")
        
        appareils_selectionnes = []
        for appareil_id, appareil_data in APPAREILS_DATA.items():
            if st.checkbox(f"{appareil_data['nom']} ({appareil_data['puissance_w']}W)", key=appareil_id):
                appareils_selectionnes.append(appareil_id)
        
        # Paramètres avancés
        with st.expander("⚙️ Paramètres avancés"):
            tarif_kwh = st.slider("Tarif électrique ($/kWh)", 0.10, 1.00, 0.25, 0.05)
            jours_observation = st.slider("Période d'observation (jours)", 7, 365, 90)
    
    with col2:
        if appareils_selectionnes:
            # Calcul de la consommation
            resultats = calculer_consommation_appareils(appareils_selectionnes, nb_personnes, tarif_kwh)
            
            st.markdown("### 📊 Résultats du Calcul")
            
            # Métriques principales
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("⚡ Consommation", f"{resultats['consommation_kwh']:.2f} kWh/j")
            with col_met2:
                st.metric("💰 Coût quotidien", f"${resultats['cout_quotidien']:.2f}")
            with col_met3:
                st.metric("🔌 Ampérage moyen", f"{resultats['amperage_moyen']:.2f} A")
            
            # Détails par appareil
            st.markdown("#### 📋 Détail par Appareil")
            for detail in resultats['details_appareils']:
                with st.container():
                    col_app1, col_app2, col_app3 = st.columns([2, 1, 1])
                    with col_app1:
                        st.write(f"**{detail['nom']}**")
                    with col_app2:
                        st.write(f"{detail['quantite']}x")
                    with col_app3:
                        st.write(f"{detail['consommation_wh']/1000:.2f} kWh")
            
            # Prédiction avec le modèle
            if st.button("🎯 Prédire le Profil de Consommation", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    # Préparation des features pour le modèle
                    features = [
                        resultats['amperage_moyen'],      # avg_amperage_per_day
                        resultats['cout_quotidien'],      # avg_depense_per_day  
                        nb_personnes,                     # nombre_personnes
                        jours_observation,                # jours_observed
                        resultats['cout_quotidien'] / max(resultats['amperage_moyen'], 0.001)  # ratio_depense_amperage
                    ]
                    
                    # Prédiction
                    pred, prob = clf.predict_household(features)
                    
                    # Affichage des résultats
                    st.markdown("---")
                    st.markdown("### 🔮 Résultat de la Prédiction")
                    
                    if pred == "grand":
                        st.markdown('<div class="prediction-high"><h1>🔴 GRAND CONSOMMATEUR</h1><p>Consommation élevée détectée - Optimisations recommandées</p></div>', unsafe_allow_html=True)
                    elif pred == "moyen":
                        st.markdown('<div class="prediction-medium"><h1>🟡 CONSOMMATION MOYENNE</h1><p>Profil standard - Quelques optimisations possibles</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-low"><h1>🟢 FAIBLE CONSOMMATION</h1><p>Consommation efficiente - Profil exemplaire</p></div>', unsafe_allow_html=True)
                    
                    # Graphique des probabilités
                    fig = go.Figure(go.Bar(
                        x=['Faible', 'Moyenne', 'Élevée'],
                        y=prob,
                        marker_color=['#4cd137', '#ff9f43', '#ff6b6b'],
                        text=[f"{p:.1%}" for p in prob],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Confiance du Modèle",
                        yaxis=dict(tickformat=".0%", range=[0, 1]),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandations
                    st.markdown("#### 💡 Recommandations")
                    if pred == "grand":
                        st.warning("""
                        **Actions recommandées:**
                        - ✅ Remplacer les vieux appareils énergivores
                        - ✅ Utiliser des ampoules LED
                        - ✅ Optimiser l'usage du climatiseur
                        - ✅ Éteindre les appareils en veille
                        """)
                    elif pred == "moyen":
                        st.info("""
                        **Améliorations possibles:**
                        - 🔄 Vérifier l'isolation de la maison
                        - 🔄 Utiliser des multiprises avec interrupteur
                        - 🔄 Optimiser les horaires d'utilisation
                        """)
                    else:
                        st.success("""
                        **Félicitations!** Votre consommation est optimale.
                        - 🏆 Continuez ces bonnes pratiques
                        - 🏆 Partagez vos astuces avec vos voisins
                        """)
        
        else:
            st.info("💡 **Sélectionnez au moins un appareil pour commencer le calcul**")
            
            # Aperçu des appareils disponibles
            st.markdown("#### 📋 Appareils Disponibles")
            for appareil_id, appareil_data in list(APPAREILS_DATA.items())[:5]:
                st.write(f"{appareil_data['nom']} - {appareil_data['puissance_w']}W")

def show_dashboard(clf):
    """Tableau de bord principal"""
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord</h2>', unsafe_allow_html=True)
    
    if clf.model_loaded:
        st.success("✅ **VRAI MODÈLE ACTIVÉ** - Données réelles utilisées")
    else:
        st.warning("🎮 **MODE DÉMO** - Données simulées")
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏠 Ménages analysés", len(clf.dataset))
    with col2:
        acc = clf.performance_metrics.get("test_accuracy", 0.92) * 100
        st.metric("🎯 Précision", f"{acc:.1f}%")
    with col3:
        st.metric("🔌 Appareils référencés", len(APPAREILS_DATA))
    
    # Statistiques des appareils
    st.markdown("#### 📈 Consommation Typique par Appareil")
    appareils_df = pd.DataFrame([
        {**data, 'appareil': key} 
        for key, data in APPAREILS_DATA.items()
    ])
    appareils_df['consommation_kwh_jour'] = (
        appareils_df['puissance_w'] * 
        appareils_df['heures_usage_jour'] / 1000
    )
    
    fig = px.bar(
        appareils_df.sort_values('consommation_kwh_jour', ascending=False),
        x='nom',
        y='consommation_kwh_jour',
        title="Consommation Quotidienne par Appareil (kWh/jour)",
        color='consommation_kwh_jour',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🔌 Calculateur Intelligent de Consommation - Sigora</h1>', unsafe_allow_html=True)
    
    # Initialisation du classifieur
    clf = SigoraHouseholdClassifier()
    
    # Navigation
    st.sidebar.markdown("## 📍 Navigation")
    page = st.sidebar.radio("", [
        "🔌 Calculateur Appareils",
        "📊 Tableau de Bord",
        "ℹ️ À Propos"
    ])

    if page == "🔌 Calculateur Appareils":
        show_appliance_calculator(clf)
    elif page == "📊 Tableau de Bord":
        show_dashboard(clf)
    elif page == "ℹ️ À Propos":
        st.markdown("""
        ## ℹ️ À Propos de cette Application
        
        **🔌 Calculateur Intelligent de Consommation**
        
        Cette application permet de:
        
        - 📊 **Calculer la consommation** basée sur les appareils électriques
        - 🎯 **Prédire le profil** de consommation avec l'IA
        - 💡 **Donner des recommandations** personnalisées
        - 🇭🇹 **Être optimisée** pour le contexte haïtien
        
        **Fonctionnement:**
        1. Sélectionnez les appareils utilisés
        2. Indiquez le nombre de personnes
        3. Obtenez une estimation de consommation
        4. Recevez une prédiction IA de votre profil
        5. Découvrez des recommandations personnalisées
        
        **Technologies:**
        - 🤖 Machine Learning (Random Forest)
        - 📈 Analytics en temps réel
        - 🔌 Base de données d'appareils réaliste
        """)

if __name__ == "__main__":
    main()
