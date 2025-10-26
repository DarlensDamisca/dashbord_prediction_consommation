#[file content begin]
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
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .impact-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
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
        self.load_artifacts()

    def load_artifacts(self):
        """Charger les fichiers du modèle"""
        st.sidebar.info("🔍 Chargement du modèle...")

        base_path = "./Model"
        if not os.path.exists(base_path):
            st.sidebar.error("❌ Dossier 'Model/' introuvable")
            self.setup_demo_mode()
            return

        try:
            files = os.listdir(base_path)
            st.sidebar.write(f"📁 Fichiers trouvés: {files}")

            # Modèle
            model_files = [f for f in files if f.startswith('best_model') and f.endswith('.joblib')]
            if model_files:
                self.model = joblib.load(os.path.join(base_path, model_files[0]))
                st.sidebar.success(f"✅ Modèle chargé: {model_files[0]}")
            else:
                st.sidebar.warning("⚠️ Modèle non trouvé")
            
            # Scaler
            if 'scaler.joblib' in files:
                self.scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
                st.sidebar.success("✅ Scaler chargé")

            # Encodeur
            if 'label_encoder.joblib' in files:
                self.encoder = joblib.load(os.path.join(base_path, 'label_encoder.joblib'))
                st.sidebar.success("✅ Encodeur chargé")

            # Données
            data_files = [f for f in files if f.startswith('final_results') and f.endswith('.csv')]
            if data_files:
                self.dataset = pd.read_csv(os.path.join(base_path, data_files[0]))
                st.sidebar.success(f"✅ Données chargées: {data_files[0]}")

            # Métriques
            if 'performance_metrics.json' in files:
                with open(os.path.join(base_path, 'performance_metrics.json'), 'r') as f:
                    self.performance_metrics = json.load(f)
                st.sidebar.success("✅ Métriques chargées")

            if self.model is None:
                st.sidebar.warning("⚠️ Fichiers incomplets - Mode démo activé")
                self.setup_demo_mode()

        except Exception as e:
            st.sidebar.error(f"❌ Erreur de chargement: {e}")
            self.setup_demo_mode()

    def setup_demo_mode(self):
        """Créer un modèle et des données fictives"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        np.random.seed(42)
        
        # Génération de données plus réalistes avec coordonnées géographiques
        n_samples = 1000
        demo_df = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, n_samples),
            'avg_depense_per_day': np.random.exponential(0.05, n_samples),
            'nombre_personnes': np.random.randint(2, 6, n_samples),
            'jours_observed': np.random.randint(30, 365, n_samples),
            'latitude': np.random.uniform(18.5, 20.0, n_samples),  # Couvre Haïti
            'longitude': np.random.uniform(-74.5, -72.0, n_samples),
            'zone': np.random.choice(['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Les Cayes'], n_samples)
        })
        
        demo_df['ratio_depense_amperage'] = demo_df['avg_depense_per_day'] / (demo_df['avg_amperage_per_day'] + 1e-9)
        demo_df['niveau_conso_pred'] = pd.cut(
            demo_df['avg_amperage_per_day'],
            bins=[-1, 0.5, 3, np.inf],
            labels=['petit', 'moyen', 'grand']
        )

        X = demo_df[['avg_amperage_per_day','avg_depense_per_day','nombre_personnes','jours_observed','ratio_depense_amperage']]
        y = demo_df['niveau_conso_pred']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_enc)
        self.dataset = demo_df
        st.sidebar.info("🎮 Mode démo activé")

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
            st.error(f"Erreur prédiction: {e}")
            return "moyen", [0.33, 0.34, 0.33]

    def detect_anomalies(self):
        """Détecter les consommations anormales"""
        if self.dataset is None:
            return []
        
        anomalies = []
        for idx, row in self.dataset.iterrows():
            # Seuils d'alerte basés sur la distribution des données
            if row['avg_amperage_per_day'] > 8.0:  # Seuil pour grand consommateur extrême
                anomalies.append({
                    'id': f"MEN{idx:04d}",
                    'type': 'Consommation Excessive',
                    'valeur': f"{row['avg_amperage_per_day']:.1f}A",
                    'seuil': '8.0A',
                    'zone': row.get('zone', 'Inconnue')
                })
            elif row['ratio_depense_amperage'] > 0.2:  # Ratio trop élevé
                anomalies.append({
                    'id': f"MEN{idx:04d}",
                    'type': 'Inefficacité Économique',
                    'valeur': f"Ratio {row['ratio_depense_amperage']:.3f}",
                    'seuil': '0.200',
                    'zone': row.get('zone', 'Inconnue')
                })
        
        return anomalies

# ==============================
# FONCTIONS DES NOUVELLES FONCTIONNALITÉS
# ==============================

def show_interactive_map(clf):
    """🎯 FONCTIONNALITÉ 1: Carte Interactive des Ménages"""
    st.markdown('<h2 class="sub-header">🗺️ Carte Interactive des Consommations</h2>', unsafe_allow_html=True)
    
    if clf.dataset is None or 'latitude' not in clf.dataset.columns:
        st.warning("📍 Données géographiques non disponibles en mode démo")
        # Créer des données géographiques simulées
        temp_df = clf.dataset.copy()
        temp_df['latitude'] = np.random.uniform(18.5, 20.0, len(temp_df))
        temp_df['longitude'] = np.random.uniform(-74.5, -72.0, len(temp_df))
    else:
        temp_df = clf.dataset
    
    # Sélecteur de type de visualisation
    viz_type = st.radio("Type de visualisation:", ["Points Colorés", "Heatmap de Densité"])
    
    if viz_type == "Points Colorés":
        fig = px.scatter_mapbox(temp_df, 
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
                               zoom=7,
                               height=600,
                               title="Répartition Géographique des Ménages")
    else:
        # Heatmap
        fig = px.density_mapbox(temp_df, 
                               lat="latitude", 
                               lon="longitude",
                               z='avg_amperage_per_day',
                               radius=20,
                               zoom=7,
                               height=600,
                               title="Heatmap de la Consommation Électrique")
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def show_impact_simulator(clf):
    """🎯 FONCTIONNALITÉ 2: Simulateur d'Impact Économique"""
    st.markdown('<h2 class="sub-header">💰 Simulateur d\'Économies Potentielles</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">💡 Choisissez un profil et des interventions pour voir leur impact</div>', unsafe_allow_html=True)
        
        menage_type = st.selectbox(
            "👨‍👩‍👧‍👦 Profil du Ménage",
            ["grand", "moyen", "petit"],
            format_func=lambda x: {"grand": "🔴 Grand Consommateur", "moyen": "🟡 Consommation Moyenne", "petit": "🟢 Faible Consommation"}[x]
        )
        
        interventions = st.multiselect(
            "🛠️ Interventions Proposées",
            ["Compteur intelligent", "Éclairage LED", "Électroménager efficace", "Sensibilisation", "Tarification incitative"],
            default=["Compteur intelligent", "Éclairage LED"]
        )
    
    with col2:
        # Calcul des économies basé sur le profil et les interventions
        economie_base = {"petit": 50, "moyen": 120, "grand": 300}[menage_type]
        multiplicateur = 1.0
        
        if "Compteur intelligent" in interventions:
            multiplicateur += 0.3
        if "Éclairage LED" in interventions:
            multiplicateur += 0.2
        if "Électroménager efficace" in interventions:
            multiplicateur += 0.4
        if "Sensibilisation" in interventions:
            multiplicateur += 0.1
        if "Tarification incitative" in interventions:
            multiplicateur += 0.25
        
        economie_totale = economie_base * multiplicateur
        
        st.markdown(f'''
        <div class="impact-card">
            <h3>📈 Impact Économique Annuel</h3>
            <h1>${economie_totale:.0f}</h1>
            <p>Économies potentielles par ménage</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Impact à l'échelle nationale
        menages_impactes = st.slider("Nombre de ménages impactés", 100, 10000, 1000)
        impact_national = economie_totale * menages_impactes
        
        st.metric("🌍 Impact National Annuel", f"${impact_national:,.0f}")

def show_real_time_alerts(clf):
    """🎯 FONCTIONNALITÉ 5: Alertes Temps Réel"""
    st.markdown('<h2 class="sub-header">🚨 Alertes Consommation Anormale</h2>', unsafe_allow_html=True)
    
    if st.button("🔄 Scanner les Anomalies", use_container_width=True):
        with st.spinner("🔍 Analyse des consommations en cours..."):
            anomalies = clf.detect_anomalies()
            
            if not anomalies:
                st.success("🎉 Aucune anomalie détectée - Toutes les consommations sont normales !")
            else:
                st.error(f"⚠️ {len(anomalies)} anomalies détectées")
                
                for anomaly in anomalies[:10]:  # Limiter à 10 affichages
                    st.markdown(f'''
                    <div class="alert-box">
                        <strong>{anomaly['id']}</strong> - {anomaly['type']}<br>
                        📊 Valeur: {anomaly['valeur']} | 🎯 Seuil: {anomaly['seuil']}<br>
                        📍 Zone: {anomaly['zone']}
                    </div>
                    ''', unsafe_allow_html=True)
                
                if len(anomalies) > 10:
                    st.info(f"💡 ... et {len(anomalies) - 10} autres anomalies. Contactez l'administrateur.")

def show_3d_clusters(clf):
    """🎯 FONCTIONNALITÉ 6: Visualisation 3D des Clusters"""
    st.markdown('<h2 class="sub-header">🔮 Visualisation 3D des Profils de Consommation</h2>', unsafe_allow_html=True)
    
    if clf.dataset is None:
        st.warning("Données non disponibles pour la visualisation 3D")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🎛️ Paramètres 3D")
        x_axis = st.selectbox("Axe X", ['avg_amperage_per_day', 'avg_depense_per_day', 'nombre_personnes', 'jours_observed'])
        y_axis = st.selectbox("Axe Y", ['avg_depense_per_day', 'avg_amperage_per_day', 'nombre_personnes', 'jours_observed'])
        z_axis = st.selectbox("Axe Z", ['nombre_personnes', 'avg_amperage_per_day', 'avg_depense_per_day', 'jours_observed'])
        
        st.markdown("---")
        st.info("💡 **Conseil:** Faites tourner la vue 3D avec votre souris !")
    
    with col2:
        fig = px.scatter_3d(clf.dataset,
                           x=x_axis,
                           y=y_axis, 
                           z=z_axis,
                           color='niveau_conso_pred',
                           color_discrete_map={
                               'petit': '#4cd137',
                               'moyen': '#ff9f43',
                               'grand': '#ff6b6b'
                           },
                           hover_data=['ratio_depense_amperage'],
                           title="Clusters 3D des Profils de Consommation",
                           height=600)
        
        fig.update_traces(marker=dict(size=5),
                         selector=dict(mode='markers'))
        
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# PAGES EXISTANTES
# ==============================

def show_dashboard(clf):
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord Principal</h2>', unsafe_allow_html=True)
    if clf.dataset is None:
        st.warning("Aucune donnée disponible")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏠 Ménages analysés", len(clf.dataset))
    with col2:
        acc = clf.performance_metrics.get("test_accuracy", 0.95) * 100 if clf.performance_metrics else 95.6
        st.metric("🎯 Précision du modèle", f"{acc:.1f}%")
    with col3:
        st.metric("🔴 Grands consommateurs", (clf.dataset["niveau_conso_pred"]=="grand").sum())
    with col4:
        st.metric("📍 Zones couvertes", clf.dataset["zone"].nunique() if "zone" in clf.dataset else 4)

    col_left, col_right = st.columns(2)
    with col_left:
        dist = clf.dataset["niveau_conso_pred"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, hole=0.4,
                     color=dist.index, color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "zone" in clf.dataset.columns:
            zone_data = clf.dataset.groupby("zone")["niveau_conso_pred"].value_counts().unstack().fillna(0)
            fig = px.bar(zone_data, barmode="stack", color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)

def show_prediction(clf):
    st.markdown('<h2 class="sub-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        avg_amperage = st.slider("Ampérage moyen (A)", 0.0, 50.0, 2.5)
        avg_depense = st.slider("Dépense moyenne ($)", 0.0, 2.0, 0.15)
        nb_personnes = st.number_input("Nombre de personnes", 1, 10, 4)
    with col2:
        jours = st.slider("Jours observés", 1, 365, 90)
        ratio = st.slider("Ratio dépense/ampérage", 0.0, 0.5, 0.06)

    if st.button("🎯 Lancer la Prédiction", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        st.markdown("---")
        if pred == "grand":
            st.markdown('<div class="prediction-high"><h1>🔴 GRAND CONSOMMATEUR</h1></div>', unsafe_allow_html=True)
        elif pred == "moyen":
            st.markdown('<div class="prediction-medium"><h1>🟡 CONSOMMATION MOYENNE</h1></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low"><h1>🟢 FAIBLE CONSOMMATION</h1></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=['Faible','Moyenne','Élevée'], y=prob,
            marker_color=['#4cd137','#ff9f43','#ff6b6b'],
            text=[f"{p:.1%}" for p in prob], textposition='auto'
        ))
        fig.update_layout(title="Probabilités de classification", yaxis=dict(tickformat=".0%", range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

def show_new_data_prediction(clf):
    st.markdown('<h2 class="sub-header">📁 Prédictions sur Nouvelles Données</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier importé : {uploaded_file.name}")

        required = ['avg_amperage_per_day','avg_depense_per_day','nombre_personnes','jours_observed','ratio_depense_amperage']
        if not all(col in new_data.columns for col in required):
            st.error(f"❌ Le fichier doit contenir : {required}")
            return

        with st.spinner("⏳ Prédiction en cours..."):
            X_scaled = clf.scaler.transform(new_data[required])
            preds = clf.model.predict(X_scaled)
            labels = clf.encoder.inverse_transform(preds)
            new_data['niveau_conso_pred'] = labels

        st.dataframe(new_data.head(50), use_container_width=True)
        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Télécharger les résultats", csv, "predictions_sigora.csv", "text/csv")

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🏠 Classification Intelligente des Ménages Haïtiens</h1>', unsafe_allow_html=True)
    clf = SigoraHouseholdClassifier()

    page = st.sidebar.radio("Navigation", [
        "🏠 Tableau de Bord",
        "🔮 Prédiction Temps Réel", 
        "📁 Nouvelles Données",
        "🗺️ Carte Interactive",
        "💰 Simulateur d'Impact",
        "🚨 Alertes Temps Réel",
        "🔮 Visualisation 3D"
    ])

    if page == "🏠 Tableau de Bord":
        show_dashboard(clf)
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction(clf)
    elif page == "📁 Nouvelles Données":
        show_new_data_prediction(clf)
    elif page == "🗺️ Carte Interactive":
        show_interactive_map(clf)
    elif page == "💰 Simulateur d'Impact":
        show_impact_simulator(clf)
    elif page == "🚨 Alertes Temps Réel":
        show_real_time_alerts(clf)
    elif page == "🔮 Visualisation 3D":
        show_3d_clusters(clf)

if __name__ == "__main__":
    main()
[file content end]
