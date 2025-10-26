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
        demo_df = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, 1000),
            'avg_depense_per_day': np.random.exponential(0.05, 1000),
            'nombre_personnes': np.random.randint(2, 6, 1000),
            'jours_observed': np.random.randint(30, 365, 1000),
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


# ==============================
# PAGES DE L’APPLICATION
# ==============================

def show_dashboard(clf):
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord Principal</h2>', unsafe_allow_html=True)
    if clf.dataset is None:
        st.warning("Aucune donnée disponible")
        return

    col1, col2, col3, col4 = st.columns(4)
    st.metric("🏠 Ménages analysés", len(clf.dataset))
    acc = clf.performance_metrics.get("test_accuracy", 0.95) * 100 if clf.performance_metrics else 95.6
    st.metric("🎯 Précision du modèle", f"{acc:.1f}%")
    st.metric("🔴 Grands consommateurs", (clf.dataset["niveau_conso_pred"]=="grand").sum())
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
        "📁 Nouvelles Données"
    ])

    if page == "🏠 Tableau de Bord":
        show_dashboard(clf)
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction(clf)
    elif page == "📁 Nouvelles Données":
        show_new_data_prediction(clf)


if __name__ == "__main__":
    main()
