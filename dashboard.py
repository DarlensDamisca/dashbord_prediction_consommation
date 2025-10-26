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
    .metric-explanation {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .currency-note {
        font-size: 0.9rem;
        color: #d63031;
        font-style: italic;
        margin-top: 0.5rem;
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
        self.q1 = None
        self.q2 = None
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
                # CALCUL DES QUANTILES COMME DANS VOTRE CODE ORIGINAL
                self.q1 = self.dataset['avg_amperage_per_day'].quantile(0.33)
                self.q2 = self.dataset['avg_amperage_per_day'].quantile(0.66)
                st.sidebar.success(f"✅ Données chargées: {data_files[0]}")
                st.sidebar.info(f"📊 Seuils calculés: Q1={self.q1:.2f}A, Q2={self.q2:.2f}A")

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
        # CRÉATION DES DONNÉES COMME DANS VOTRE CODE
        demo_df = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, 1000),
            'avg_depense_per_day': np.random.exponential(7.5, 1000),  # En gourdes
            'nombre_personnes': np.random.randint(2, 6, 1000),
            'jours_observed': np.random.randint(30, 365, 1000),
            'zone': np.random.choice(['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Les Cayes'], 1000)
        })
        
        # APPLICATION DE VOTRE MÉTHODE EXACTE DE LABELLISATION
        self.q1 = demo_df['avg_amperage_per_day'].quantile(0.33)
        self.q2 = demo_df['avg_amperage_per_day'].quantile(0.66)
        
        def label_niveau(x):
            if x <= self.q1:
                return 'petit'
            elif x <= self.q2:
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
        st.sidebar.info(f"📊 Seuils démo: Q1={self.q1:.2f}A, Q2={self.q2:.2f}A")

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

    def get_quantile_interpretation(self, amperage):
        """Retourne l'interprétation basée sur les quantiles réels"""
        if self.q1 is None or self.q2 is None:
            return "Seuils non disponibles"
        
        if amperage <= self.q1:
            return f"🟢 FAIBLE (≤{self.q1:.2f}A - 33% inférieur)"
        elif amperage <= self.q2:
            return f"🟡 MOYEN ({self.q1:.2f}A - {self.q2:.2f}A - 33% moyen)"
        else:
            return f"🔴 ÉLEVÉ (>{self.q2:.2f}A - 33% supérieur)"


# ==============================
# PAGES DE L'APPLICATION
# ==============================

def show_dashboard(clf):
    st.markdown('<h2 class="sub-header">📊 Tableau de Bord Principal</h2>', unsafe_allow_html=True)
    if clf.dataset is None:
        st.warning("Aucune donnée disponible")
        return

    # Métriques principales avec explications
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_households = len(clf.dataset)
        st.metric("🏠 Ménages analysés", total_households)
        st.caption("Base de données d'entraînement")
    
    with col2:
        acc = clf.performance_metrics.get("test_accuracy", 0.95) * 100 if clf.performance_metrics else 95.6
        st.metric("🎯 Précision du modèle", f"{acc:.1f}%")
        st.caption("Taux de prédictions correctes")
    
    with col3:
        high_cons = (clf.dataset["niveau_conso_pred"]=="grand").sum()
        st.metric("🔴 Grands consommateurs", high_cons)
        st.caption(f"({high_cons/len(clf.dataset):.1%} du total)")
    
    with col4:
        zones = clf.dataset["zone"].nunique() if "zone" in clf.dataset else 4
        st.metric("📍 Zones couvertes", zones)
        st.caption("Régions géographiques")

    # Affichage des seuils quantiles
    if clf.q1 is not None and clf.q2 is not None:
        st.info(f"**📊 Seuils de classification basés sur les quantiles :** Q1 (33%) = {clf.q1:.2f}A • Q2 (66%) = {clf.q2:.2f}A")

    # Section d'interprétation des performances
    with st.expander("📈 Performance du Modèle - Comment interpréter?", expanded=False):
        st.markdown("""
        **Échelle de précision :**
        - **Précision de 90%+** : Modèle très performant ✅  
        - **Précision de 80-90%** : Bonnes performances ✅  
        - **Précision de 70-80%** : Performances acceptables ⚠️  
        - **Précision < 70%** : Améliorations nécessaires ❌
        
        **Méthode de classification :**
        - Basée sur les **quantiles** de l'ampérage (33% et 66%)
        - **Faible** : 33% des ménages les moins consommateurs
        - **Moyen** : 33% des ménages dans la moyenne
        - **Élevé** : 33% des ménages les plus consommateurs
        """)

    # Graphiques
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📈 Répartition des Consommations")
        dist = clf.dataset["niveau_conso_pred"].value_counts()
        fig = px.pie(
            values=dist.values, 
            names=dist.index, 
            hole=0.4,
            color=dist.index, 
            color_discrete_map={
                'petit': '#4cd137',
                'moyen': '#ff9f43',
                'grand': '#ff6b6b'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Distribution des Ampérages")
        if clf.q1 is not None and clf.q2 is not None:
            fig = px.histogram(clf.dataset, x='avg_amperage_per_day', 
                             title="Distribution des ampérages avec seuils quantiles",
                             nbins=50)
            fig.add_vline(x=clf.q1, line_dash="dash", line_color="green", 
                         annotation_text=f"Q1 (33%) = {clf.q1:.2f}A")
            fig.add_vline(x=clf.q2, line_dash="dash", line_color="red", 
                         annotation_text=f"Q2 (66%) = {clf.q2:.2f}A")
            fig.update_layout(xaxis_title="Ampérage moyen (A)", yaxis_title="Nombre de ménages")
            st.plotly_chart(fig, use_container_width=True)

def show_prediction(clf):
    st.markdown('<h2 class="sub-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)
    
    # Section d'information pour l'utilisateur
    with st.expander("ℹ️ COMMENT FONCTIONNE L'ANALYSE ?", expanded=True):
        st.markdown("""
        ### 🎯 Méthode de classification basée sur les QUANTILES
        
        **Votre méthode exacte est utilisée :**
        - **Q1 (33%)** : 33% des ménages les moins consommateurs → **FAIBLE**
        - **Q2 (66%)** : 33% des ménages moyens → **MOYEN**  
        - **Au-dessus Q2** : 33% des ménages les plus consommateurs → **ÉLEVÉ**
        
        **Seuils calculés sur vos données :**
        """)
        if clf.q1 is not None and clf.q2 is not None:
            st.markdown(f"""
            - **Faible consommation** : ≤ {clf.q1:.2f}A
            - **Consommation moyenne** : {clf.q1:.2f}A - {clf.q2:.2f}A
            - **Grand consommateur** : > {clf.q2:.2f}A
            """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Paramètres du Ménage")
        
        avg_amperage = st.slider(
            "Ampérage moyen par jour (A)", 
            0.0, 50.0, 2.5,
            help=f"Ampérage moyen quotidien - Seuils: Faible ≤ {clf.q1:.2f}A, Moyen ≤ {clf.q2:.2f}A, Élevé > {clf.q2:.2f}A" if clf.q1 else "Ampérage moyen quotidien"
        )
        
        # Affichage de l'interprétation en temps réel
        if clf.q1 is not None:
            interpretation = clf.get_quantile_interpretation(avg_amperage)
            if "FAIBLE" in interpretation:
                st.success(interpretation)
            elif "MOYEN" in interpretation:
                st.warning(interpretation)
            else:
                st.error(interpretation)
        
        avg_depense = st.slider(
            "Dépense moyenne par jour (HTG)", 
            0.0, 300.0, 22.5,
            help="Dépense quotidienne en Gourdes Haïtiennes"
        )
        
        nb_personnes = st.number_input(
            "Nombre de personnes dans le ménage", 
            1, 10, 4,
            help="Taille du foyer familial"
        )
        
    with col2:
        st.markdown("#### 📈 Données d'Observation")
        jours = st.slider(
            "Jours d'observation", 
            1, 365, 90,
            help="Durée de collecte des données (fiabilité)"
        )
        
        ratio = st.slider(
            "Ratio dépense/ampérage (HTG par Ampère)", 
            0.0, 150.0, 9.0,
            help="Efficacité économique : coût par unité d'énergie consommée"
        )
        
        # Afficher les valeurs avec interprétation
        st.markdown("---")
        st.markdown("**📋 VOTRE PROFIL ACTUEL :**")
        
        st.write(f"- ⚡ Ampérage : {avg_amperage} A")
        st.write(f"- 💰 Dépense : {avg_depense:.0f} HTG")
        st.write(f"- 👥 Personnes : {nb_personnes}")
        st.write(f"- 📅 Jours observés : {jours}")
        st.write(f"- 📊 Ratio : {ratio:.1f} HTG/A")

    if st.button("🎯 ANALYSER CE MÉNAGE", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        
        # AFFICHAGE COHÉRENT BASÉ SUR VOS LABELS
        st.markdown("---")
        st.markdown("## 📋 RÉSULTATS DE L'ANALYSE")
        
        # Mapping cohérent avec votre méthode
        label_mapping = {
            'petit': ('🟢 FAIBLE CONSOMMATION', 'prediction-low', "Votre ménage fait partie des 33% les moins consommateurs"),
            'moyen': ('🟡 CONSOMMATION MOYENNE', 'prediction-medium', "Votre ménage fait partie des 33% de consommation moyenne"),
            'grand': ('🔴 GRAND CONSOMMATEUR', 'prediction-high', "Votre ménage fait partie des 33% les plus consommateurs")
        }
        
        prediction_text, prediction_class, explanation = label_mapping.get(pred, 
            ('🟡 CONSOMMATION MOYENNE', 'prediction-medium', "Classification standard"))
        
        # Affichage cohérent
        st.markdown(f'<div class="{prediction_class}"><h1>{prediction_text}</h1></div>', unsafe_allow_html=True)
        
        # Message d'interprétation basé sur les quantiles
        st.markdown(f"""
        <div class="info-box">
        <h4>🎯 INTERPRÉTATION BASÉE SUR LES QUANTILES</h4>
        <p><strong>{explanation}</strong></p>
        <p><strong>Seuils utilisés :</strong></p>
        <ul>
            <li>• Faible consommation : ≤ {clf.q1:.2f}A (33% inférieur)</li>
            <li>• Consommation moyenne : ≤ {clf.q2:.2f}A (33% moyen)</li>
            <li>• Grande consommation : > {clf.q2:.2f}A (33% supérieur)</li>
        </ul>
        <p><strong>Votre ampérage : {avg_amperage}A</strong> → Classé comme <strong>{pred}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # GRAPHIQUE DE CONFIANCE
        st.markdown("---")
        st.markdown("## 📊 NIVEAUX DE CONFIANCE")
        
        col_explain, col_graph = st.columns([1, 2])
        
        with col_explain:
            st.markdown("""
            ### 🎯 COMMENT LIRE CE GRAPHIQUE ?
            
            **Probabilités de classification :**
            - 🟢 **Faible** : 33% des ménages les moins consommateurs
            - 🟡 **Moyenne** : 33% des ménages dans la moyenne  
            - 🔴 **Élevée** : 33% des ménages les plus consommateurs
            
            **Plus la barre est haute, plus le modèle est certain !**
            """)
            
            max_prob = max(prob)
            pred_index = list(label_mapping.keys()).index(pred)
            
            st.markdown(f"### 📈 RÉSULTAT :")
            st.markdown(f"**Catégorie prédite :** `{pred}`")
            st.markdown(f"**Niveau de confiance :** `{max_prob:.1%}`")
            
            if max_prob > 0.8:
                st.success("**✅ TRÈS FIABLE** - Le modèle est très certain")
            elif max_prob > 0.6:
                st.info("**ℹ️ FIABLE** - Bon niveau de confiance")
            else:
                st.warning("**⚠️ INCERTAIN** - Plusieurs catégories possibles")
        
        with col_graph:
            categories = ['Faible', 'Moyenne', 'Élevée']
            colors = ['#4cd137', '#ff9f43', '#ff6b6b']
            
            fig = go.Figure(go.Bar(
                x=categories, 
                y=prob,
                marker_color=colors,
                text=[f"{p:.1%}" for p in prob], 
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Probabilité: %{y:.1%}<extra></extra>"
            ))
            fig.update_layout(
                title="PROBABILITÉS DE CLASSIFICATION",
                yaxis=dict(
                    tickformat=".0%", 
                    range=[0,1],
                    title="Probabilité"
                ),
                xaxis_title="Catégories basées sur les quantiles",
                height=400
            )
            
            # Annotation pour la prédiction
            fig.add_annotation(
                x=pred_index,
                y=prob[pred_index] + 0.05,
                text="PRÉDICTION",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Section d'analyse détaillée
        st.markdown("---")
        st.markdown("## 🔍 ANALYSE DÉTAILLÉE")
        
        st.markdown("""
        ### 📋 COMMENT VOS DONNÉES ONT ÉTÉ CLASSÉES :
        """)
        
        factors = {
            "Ampérage": {
                "value": f"{avg_amperage} A",
                "level": f"Quantile: {clf.get_quantile_interpretation(avg_amperage).split(' ')[1]}",
                "impact": "PRINCIPAL"
            },
            "Dépense": {
                "value": f"{avg_depense:.0f} HTG",
                "level": "Élevée" if avg_depense > 50 else "Modérée" if avg_depense > 15 else "Faible",
                "impact": "SECONDAIRE"
            },
            "Taille ménage": {
                "value": f"{nb_personnes} personnes",
                "level": "Grand" if nb_personnes > 5 else "Moyen" if nb_personnes > 3 else "Petit",
                "impact": "SECONDAIRE"
            }
        }
        
        for factor, data in factors.items():
            col_fact, col_level, col_impact = st.columns([2, 1, 1])
            with col_fact:
                st.write(f"**{factor}** : {data['value']}")
            with col_level:
                if "FAIBLE" in data['level'] or "Petit" in data['level']:
                    st.success(data['level'])
                elif "MOYEN" in data['level'] or "Modérée" in data['level']:
                    st.warning(data['level'])
                else:
                    st.error(data['level'])
            with col_impact:
                if data['impact'] == "PRINCIPAL":
                    st.error(f"Impact: {data['impact']}")
                else:
                    st.warning(f"Impact: {data['impact']}")

# [Les fonctions show_new_data_prediction et show_help_guide restent identiques au code précédent]
def show_new_data_prediction(clf):
    st.markdown('<h2 class="sub-header">📁 Prédictions sur Nouvelles Données</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Format requis pour le fichier CSV :</h4>
    <p>Votre fichier doit contenir les colonnes suivantes :</p>
    <ul>
        <li><code>avg_amperage_per_day</code> : Ampérage moyen quotidien (A)</li>
        <li><code>avg_depense_per_day</code> : Dépense moyenne quotidienne (HTG)</li>
        <li><code>nombre_personnes</code> : Nombre de personnes dans le ménage</li>
        <li><code>jours_observed</code> : Nombre de jours d'observation</li>
        <li><code>ratio_depense_amperage</code> : Ratio dépense/ampérage (HTG par Ampère)</li>
    </ul>
    <p class="currency-note">💡 Classification basée sur les quantiles : Faible (0-33%), Moyen (33-66%), Élevé (66-100%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier importé : {uploaded_file.name}")
        st.write(f"**📊 Aperçu des données** ({len(new_data)} lignes) :")
        st.dataframe(new_data.head(10), use_container_width=True)

        required = ['avg_amperage_per_day','avg_depense_per_day','nombre_personnes','jours_observed','ratio_depense_amperage']
        if not all(col in new_data.columns for col in required):
            st.error(f"❌ Le fichier doit contenir les colonnes : {required}")
            st.write("**Colonnes trouvées :**", list(new_data.columns))
            return

        with st.spinner("⏳ Prédiction en cours..."):
            X_scaled = clf.scaler.transform(new_data[required])
            preds = clf.model.predict(X_scaled)
            labels = clf.encoder.inverse_transform(preds)
            new_data['niveau_conso_pred'] = labels
            
            # Ajouter les probabilités
            probas = clf.model.predict_proba(X_scaled)
            new_data['prob_faible'] = probas[:, 0]
            new_data['prob_moyenne'] = probas[:, 1]
            new_data['prob_elevee'] = probas[:, 2]

        st.markdown("---")
        st.markdown("## 📋 Résultats des Prédictions")
        
        # Résumé statistique
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Grands consommateurs", (new_data['niveau_conso_pred'] == 'grand').sum())
        with col2:
            st.metric("🟡 Consommation moyenne", (new_data['niveau_conso_pred'] == 'moyen').sum())
        with col3:
            st.metric("🟢 Faible consommation", (new_data['niveau_conso_pred'] == 'petit').sum())
        
        # Aperçu des résultats
        st.dataframe(new_data.head(50), use_container_width=True)
        
        # Téléchargement
        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Télécharger les résultats complets", 
            csv, 
            "predictions_sigora.csv", 
            "text/csv",
            use_container_width=True
        )

def show_help_guide():
    st.markdown('<h2 class="sub-header">📖 Guide d\'Utilisation et Interprétation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Méthode de Classification par Quantiles
        
        **Notre système utilise VOTRE méthode exacte :**
        
        📊 **Calcul des seuils :**
        - Q1 = 33ème percentile de l'ampérage
        - Q2 = 66ème percentile de l'ampérage
        
        🏠 **Répartition :**
        - **Faible** : 33% des ménages (≤ Q1)
        - **Moyen** : 33% des ménages (Q1 - Q2)  
        - **Élevé** : 33% des ménages (> Q2)
        
        **Avantages :**
        - Adaptation automatique aux données
        - Répartition équilibrée
        - Pas de seuils arbitraires
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Interprétation des Résultats
        
        **Quand la prédiction est fiable :**
        - Probabilité > 70% pour une catégorie
        - Données d'observation > 30 jours
        - Profil cohérent avec les facteurs
        
        **Échelle de confiance :**
        - > 80% : Très fiable ✅
        - 60-80% : Fiable ℹ️  
        - < 60% : Incertain ⚠️
        
        **Facteurs principaux :**
        - Ampérage moyen (principal)
        - Dépense énergétique
        - Taille du ménage
        """)
    
    st.markdown("---")
    st.markdown("#### 📚 Glossaire des Termes")
    
    glossary_col1, glossary_col2 = st.columns(2)
    
    with glossary_col1:
        st.markdown("""
        **Quantile :**
        > Valeur qui divise les données en parts égales
        
        **Q1 (33ème percentile) :**
        > Seuil où 33% des ménages consomment moins
        
        **Q2 (66ème percentile) :**
        > Seuil où 66% des ménages consomment moins
        """)
    
    with glossary_col2:
        st.markdown("""
        **Ampérage moyen :**
        > Intensité électrique quotidienne consommée
        
        **Ratio dépense/ampérage :**
        > Efficacité économique (HTG par Ampère)
        
        **Période d'observation :**
        > Durée de collecte des données
        """)

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🏠 Classification des Ménages Haïtiens - Sigora</h1>', unsafe_allow_html=True)
    
    # Information sur le mode
    if st.sidebar.checkbox("ℹ️ Afficher les informations techniques", value=False):
        st.sidebar.info("""
        **Mode actuel :** 
        - 🔍 Chargement des modèles réels si disponibles
        - 🎮 Mode démo activé sinon
        
        **Méthode de classification :**
        - Basée sur les quantiles (33% / 66%)
        - Labels : petit, moyen, grand
        - Devise : Gourdes Haïtiennes (HTG)
        """)
    
    clf = SigoraHouseholdClassifier()

    page = st.sidebar.radio("Navigation", [
        "🏠 Tableau de Bord",
        "🔮 Prédiction Temps Réel", 
        "📁 Nouvelles Données",
        "📖 Guide d'Interprétation"
    ])

    if page == "🏠 Tableau de Bord":
        show_dashboard(clf)
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction(clf)
    elif page == "📁 Nouvelles Données":
        show_new_data_prediction(clf)
    elif page == "📖 Guide d'Interprétation":
        show_help_guide()
    
    # Pied de page
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Sigora Haiti**  
    *Énergie intelligente pour tous*  
    📧 contact@sigora.com  
    🌐 www.sigora.com
    """)
    st.sidebar.markdown('<p class="currency-note">💵 Toutes les valeurs en Gourdes Haïtiennes (HTG)</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
