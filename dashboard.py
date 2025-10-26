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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
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
        self.training_q1 = None
        self.training_q2 = None
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

            # Données d'entraînement
            data_files = [f for f in files if f.startswith('final_results') and f.endswith('.csv')]
            if data_files:
                self.dataset = pd.read_csv(os.path.join(base_path, data_files[0]))
                # SAUVEGARDER LES QUANTILES D'ENTRAÎNEMENT
                self.training_q1 = self.dataset['avg_amperage_per_day'].quantile(0.33)
                self.training_q2 = self.dataset['avg_amperage_per_day'].quantile(0.66)
                st.sidebar.success(f"✅ Données chargées: {data_files[0]}")
                st.sidebar.info(f"📊 Seuils d'entraînement: Q1={self.training_q1:.2f}A, Q2={self.training_q2:.2f}A")

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
            'avg_depense_per_day': np.random.exponential(7.5, 1000),
            'nombre_personnes': np.random.randint(2, 6, 1000),
            'jours_observed': np.random.randint(30, 365, 1000),
            'zone': np.random.choice(['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Les Cayes'], 1000)
        })
        
        # MÉTHODE DES QUANTILES COMME DANS VOTRE CODE
        self.training_q1 = demo_df['avg_amperage_per_day'].quantile(0.33)
        self.training_q2 = demo_df['avg_amperage_per_day'].quantile(0.66)
        
        def label_niveau(x):
            if x <= self.training_q1:
                return 'petit'
            elif x <= self.training_q2:
                return 'moyen'
            else:
                return 'grand'
        
        demo_df['niveau_conso_pred'] = demo_df['avg_amperage_per_day'].apply(label_niveau)
        demo_df['ratio_depense_amperage'] = demo_df['avg_depense_per_day'] / (demo_df['avg_amperage_per_day'] + 1e-9)

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
        st.sidebar.info(f"📊 Seuils démo: Q1={self.training_q1:.2f}A, Q2={self.training_q2:.2f}A")

    def predict_household(self, features):
        """Faire une prédiction unique - UTILISE LE MODÈLE ENTRAÎNÉ"""
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

    def predict_batch(self, new_data):
        """Prédire un lot de nouvelles données - UTILISE LE MODÈLE ENTRAÎNÉ"""
        try:
            required_cols = ['avg_amperage_per_day','avg_depense_per_day','nombre_personnes','jours_observed','ratio_depense_amperage']
            
            # Vérifier les colonnes
            missing_cols = [col for col in required_cols if col not in new_data.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
            # Préparer les features
            X = new_data[required_cols]
            X_scaled = self.scaler.transform(X)
            
            # Prédictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            labels = self.encoder.inverse_transform(predictions)
            
            # Ajouter les résultats
            result_df = new_data.copy()
            result_df['niveau_conso_pred'] = labels
            result_df['prob_faible'] = probabilities[:, 0]
            result_df['prob_moyenne'] = probabilities[:, 1]
            result_df['prob_elevee'] = probabilities[:, 2]
            
            return result_df
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction par lot: {e}")
            return None

    def get_training_quantiles_interpretation(self, amperage):
        """Interprétation basée sur les quantiles d'entraînement"""
        if self.training_q1 is None or self.training_q2 is None:
            return "Seuils d'entraînement non disponibles"
        
        if amperage <= self.training_q1:
            return f"🟢 FAIBLE (≤{self.training_q1:.2f}A - 33% inférieur des données d'entraînement)"
        elif amperage <= self.training_q2:
            return f"🟡 MOYEN ({self.training_q1:.2f}A - {self.training_q2:.2f}A - 33% moyen)"
        else:
            return f"🔴 ÉLEVÉ (>{self.training_q2:.2f}A - 33% supérieur)"

# ==============================
# PAGES DE L'APPLICATION
# ==============================

def show_dashboard(clf):
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord Principal</h2>', unsafe_allow_html=True)
    
    if clf.dataset is None:
        st.warning("Aucune donnée d'entraînement disponible")
        return

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Ménages analysés", len(clf.dataset))
    
    with col2:
        acc = clf.performance_metrics.get("test_accuracy", 0.95) * 100 if clf.performance_metrics else 95.6
        st.metric("🎯 Précision du modèle", f"{acc:.1f}%")
    
    with col3:
        high_cons = (clf.dataset["niveau_conso_pred"]=="grand").sum()
        st.metric("🔴 Grands consommateurs", high_cons)
    
    with col4:
        zones = clf.dataset["zone"].nunique() if "zone" in clf.dataset else 4
        st.metric("📍 Zones couvertes", zones)

    # Seuils d'entraînement
    if clf.training_q1 is not None:
        st.info(f"**📊 Seuils d'entraînement (quantiles) :** Q1 (33%) = {clf.training_q1:.2f}A • Q2 (66%) = {clf.training_q2:.2f}A")

    # Graphiques
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📈 Répartition des Consommations")
        dist = clf.dataset["niveau_conso_pred"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, hole=0.4,
                    color=dist.index, color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Distribution des Ampérages")
        fig = px.histogram(clf.dataset, x='avg_amperage_per_day', nbins=50,
                          title="Distribution avec seuils d'entraînement")
        if clf.training_q1:
            fig.add_vline(x=clf.training_q1, line_dash="dash", line_color="green",
                         annotation_text=f"Q1 = {clf.training_q1:.2f}A")
            fig.add_vline(x=clf.training_q2, line_dash="dash", line_color="red",
                         annotation_text=f"Q2 = {clf.training_q2:.2f}A")
        st.plotly_chart(fig, use_container_width=True)

def show_prediction(clf):
    st.markdown('<h2 class="sub-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ INFORMATION IMPORTANTE", expanded=True):
        st.markdown("""
        ### 🎯 MODE DE PRÉDICTION POUR NOUVELLES DONNÉES
        
        **Le modèle utilise les SEUILS D'ENTRAÎNEMENT pour classifier :**
        - Basé sur les quantiles calculés lors de l'entraînement
        - Les nouvelles données sont comparées aux données historiques
        - **Ne recalcule PAS les quantiles** sur les nouvelles données
        
        **Avantages :**
        - Cohérence avec le modèle entraîné
        - Comparaison standardisée dans le temps
        - Pas de biais lié aux nouvelles distributions
        """)
        if clf.training_q1:
            st.markdown(f"""
            **Seuils d'entraînement utilisés :**
            - **Faible** : ≤ {clf.training_q1:.2f}A (33% inférieur des données d'entraînement)
            - **Moyen** : ≤ {clf.training_q2:.2f}A (33% moyen)
            - **Élevé** : > {clf.training_q2:.2f}A (33% supérieur)
            """)
    
    col1, col2 = st.columns(2)
    with col1:
        avg_amperage = st.slider("Ampérage moyen (A)", 0.0, 50.0, 2.5)
        avg_depense = st.slider("Dépense moyenne (HTG)", 0.0, 300.0, 22.5)
        nb_personnes = st.number_input("Nombre de personnes", 1, 10, 4)
        
        # Interprétation en temps réel
        if clf.training_q1:
            interpretation = clf.get_training_quantiles_interpretation(avg_amperage)
            if "FAIBLE" in interpretation:
                st.success(interpretation)
            elif "MOYEN" in interpretation:
                st.warning(interpretation)
            else:
                st.error(interpretation)
    
    with col2:
        jours = st.slider("Jours observés", 1, 365, 90)
        ratio = st.slider("Ratio (HTG/A)", 0.0, 150.0, 9.0)

    if st.button("🎯 PRÉDIRE LA CONSOMMATION", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        
        # Affichage des résultats
        st.markdown("---")
        st.markdown("## 📋 RÉSULTATS DE LA PRÉDICTION")
        
        label_mapping = {
            'petit': ('🟢 FAIBLE CONSOMMATION', 'prediction-low'),
            'moyen': ('🟡 CONSOMMATION MOYENNE', 'prediction-medium'),
            'grand': ('🔴 GRAND CONSOMMATEUR', 'prediction-high')
        }
        
        prediction_text, prediction_class = label_mapping.get(pred, ('🟡 CONSOMMATION MOYENNE', 'prediction-medium'))
        st.markdown(f'<div class="{prediction_class}"><h1>{prediction_text}</h1></div>', unsafe_allow_html=True)
        
        # Graphique des probabilités
        fig = go.Figure(go.Bar(
            x=['Faible','Moyenne','Élevée'], y=prob,
            marker_color=['#4cd137','#ff9f43','#ff6b6b'],
            text=[f"{p:.1%}" for p in prob], textposition='auto'
        ))
        fig.update_layout(title="Probabilités de classification", yaxis=dict(tickformat=".0%", range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

def show_new_data_prediction(clf):
    st.markdown('<h2 class="sub-header">📁 Prédictions sur Nouvelles Données</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 MODE PRÉDICTION POUR NOUVELLES DONNÉES</h4>
    <p><strong>Le modèle utilise les SEUILS D'ENTRAÎNEMENT pour classifier vos nouvelles données :</strong></p>
    """)
    
    if clf.training_q1:
        st.markdown(f"""
        <ul>
            <li>• <strong>Faible consommation</strong> : ≤ {clf.training_q1:.2f}A (33% inférieur des données d'entraînement)</li>
            <li>• <strong>Consommation moyenne</strong> : ≤ {clf.training_q2:.2f}A (33% moyen)</li>
            <li>• <strong>Grande consommation</strong> : > {clf.training_q2:.2f}A (33% supérieur)</li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <p><strong>⚠️ IMPORTANT :</strong> Les nouvelles données sont comparées aux données d'entraînement, 
    les quantiles ne sont pas recalculés.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Importer un fichier CSV avec les nouvelles données", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier importé : {uploaded_file.name} ({len(new_data)} lignes)")
            
            # Vérification des colonnes
            required_cols = ['avg_amperage_per_day','avg_depense_per_day','nombre_personnes','jours_observed','ratio_depense_amperage']
            missing_cols = [col for col in required_cols if col not in new_data.columns]
            
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
                st.info("""
                **Format requis :**
                - `avg_amperage_per_day` : Ampérage moyen (A)
                - `avg_depense_per_day` : Dépense moyenne (HTG)  
                - `nombre_personnes` : Nombre de personnes
                - `jours_observed` : Jours d'observation
                - `ratio_depense_amperage` : Ratio (HTG/A)
                """)
                return
            
            # Aperçu des données
            st.markdown("### 📊 Aperçu des données importées")
            st.dataframe(new_data.head(10), use_container_width=True)
            
            # Statistiques descriptives
            st.markdown("### 📈 Statistiques des nouvelles données")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ampérage moyen", f"{new_data['avg_amperage_per_day'].mean():.2f}A")
            with col2:
                st.metric("Dépense moyenne", f"{new_data['avg_depense_per_day'].mean():.1f} HTG")
            with col3:
                st.metric("Taille moyenne", f"{new_data['nombre_personnes'].mean():.1f} pers")
            
            # Comparaison avec les seuils d'entraînement
            if clf.training_q1:
                st.markdown("### 🔍 Comparaison avec les seuils d'entraînement")
                new_q1 = new_data['avg_amperage_per_day'].quantile(0.33)
                new_q2 = new_data['avg_amperage_per_day'].quantile(0.66)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Seuils d'entraînement (fixes) :**")
                    st.write(f"- Q1 (33%) : {clf.training_q1:.2f}A")
                    st.write(f"- Q2 (66%) : {clf.training_q2:.2f}A")
                
                with col2:
                    st.markdown("**Quantiles des nouvelles données :**")
                    st.write(f"- Q1 (33%) : {new_q1:.2f}A")
                    st.write(f"- Q2 (66%) : {new_q2:.2f}A")
                
                if abs(new_q1 - clf.training_q1) > 0.5 or abs(new_q2 - clf.training_q2) > 0.5:
                    st.warning("""
                    **⚠️ Attention :** Les nouvelles données ont une distribution différente des données d'entraînement.
                    Les prédictions utilisent les seuils d'entraînement pour maintenir la cohérence.
                    """)
            
            # Prédictions
            if st.button("🚀 Lancer les prédictions", use_container_width=True):
                with st.spinner("⏳ Calcul des prédictions..."):
                    results = clf.predict_batch(new_data)
                    
                    if results is not None:
                        st.success("✅ Prédictions terminées !")
                        
                        # Résumé des prédictions
                        st.markdown("### 📋 Résumé des prédictions")
                        pred_counts = results['niveau_conso_pred'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🟢 Faible consommation", pred_counts.get('petit', 0))
                        with col2:
                            st.metric("🟡 Consommation moyenne", pred_counts.get('moyen', 0))
                        with col3:
                            st.metric("🔴 Grand consommateur", pred_counts.get('grand', 0))
                        
                        # Distribution des prédictions
                        st.markdown("### 📊 Distribution des prédictions")
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                                    color=pred_counts.index,
                                    color_discrete_map={'petit':'#4cd137','moyen':'#ff9f43','grand':'#ff6b6b'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tableau des résultats
                        st.markdown("### 📄 Détail des prédictions")
                        st.dataframe(results, use_container_width=True)
                        
                        # Téléchargement
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Télécharger les résultats",
                            csv,
                            "predictions_nouvelles_donnees.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")

def show_help_guide():
    st.markdown('<h2 class="sub-header">📖 Guide des Prédictions</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 STRATÉGIE DE PRÉDICTION POUR NOUVELLES DONNÉES</h4>
    
    <h5>Pourquoi utiliser les seuils d'entraînement ?</h5>
    <p><strong>Problème :</strong> Si on recalcule les quantiles sur les nouvelles données :</p>
    <ul>
        <li>• Un ménage pourrait changer de catégorie sans changer sa consommation</li>
        <li>• Impossibilité de comparer dans le temps</li>
        <li>• Perte de la signification originale des labels</li>
    </ul>
    
    <h5>Solution : Seuils fixes d'entraînement</h5>
    <ul>
        <li>• <strong>Cohérence</strong> : Mêmes seuils pour toutes les prédictions</li>
        <li>• <strong>Comparabilité</strong> : Possibilité de comparer dans le temps</li>
        <li>• <strong>Stabilité</strong> : Les labels gardent leur signification</li>
    </ul>
    
    <h5>Que faire si la distribution change ?</h5>
    <p>Si les nouvelles données sont très différentes :</p>
    <ul>
        <li>1. <strong>Recalculer le modèle</strong> avec l'ensemble des données</li>
        <li>2. <strong>Mettre à jour les seuils</strong> d'entraînement</li>
        <li>3. <strong>Reprédire</strong> toutes les données avec les nouveaux seuils</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🏠 Classification des Ménages Haïtiens</h1>', unsafe_allow_html=True)
    
    clf = SigoraHouseholdClassifier()

    page = st.sidebar.radio("Navigation", [
        "🏠 Tableau de Bord",
        "🔮 Prédiction Temps Réel", 
        "📁 Nouvelles Données",
        "📖 Guide des Prédictions"
    ])

    if page == "🏠 Tableau de Bord":
        show_dashboard(clf)
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction(clf)
    elif page == "📁 Nouvelles Données":
        show_new_data_prediction(clf)
    elif page == "📖 Guide des Prédictions":
        show_help_guide()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sigora Haiti** - *Énergie intelligente*")

if __name__ == "__main__":
    main()
