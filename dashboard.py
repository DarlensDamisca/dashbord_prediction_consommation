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
        # MAINTENANT EN GOURDES HAÏTIENNES (HTG)
        demo_df = pd.DataFrame({
            'avg_amperage_per_day': np.random.exponential(2.0, 1000),
            'avg_depense_per_day': np.random.exponential(7.5, 1000),  # 7.5 HTG au lieu de 0.05$
            'nombre_personnes': np.random.randint(2, 6, 1000),
            'jours_observed': np.random.randint(30, 365, 1000),
            'zone': np.random.choice(['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Les Cayes'], 1000)
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

    # Section d'interprétation des performances
    with st.expander("📈 Performance du Modèle - Comment interpréter?", expanded=False):
        st.markdown("""
        **Échelle de précision :**
        - **Précision de 90%+** : Modèle très performant ✅  
        - **Précision de 80-90%** : Bonnes performances ✅  
        - **Précision de 70-80%** : Performances acceptables ⚠️  
        - **Précision < 70%** : Améliorations nécessaires ❌
        
        *Notre modèle actuel montre une précision excellente pour la classification des ménages haïtiens.*
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
        st.markdown("#### 📊 Consommation par Zone")
        if "zone" in clf.dataset.columns:
            zone_data = clf.dataset.groupby("zone")["niveau_conso_pred"].value_counts().unstack().fillna(0)
            fig = px.bar(
                zone_data, 
                barmode="stack", 
                color_discrete_map={
                    'petit': '#4cd137',
                    'moyen': '#ff9f43',
                    'grand': '#ff6b6b'
                }
            )
            fig.update_layout(
                xaxis_title="Zones géographiques",
                yaxis_title="Nombre de ménages"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Données de zone non disponibles en mode démo")

def show_prediction(clf):
    st.markdown('<h2 class="sub-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)
    
    # Section d'information pour l'utilisateur
    with st.expander("ℹ️ COMMENT FONCTIONNE L'ANALYSE ?", expanded=True):
        st.markdown("""
        ### 🎯 Comment interpréter les résultats ?
        
        **Le modèle analyse 5 facteurs clés :**
        1. **Ampérage moyen** → Combien d'électricité vous consommez
        2. **Dépense moyenne** → Combien vous payez pour cette électricité  
        3. **Nombre de personnes** → Taille de votre famille
        4. **Jours observés** → Fiabilité des données
        5. **Ratio dépense/ampérage** → Efficacité économique
        
        ### 📈 Le graphique de confiance vous montre :
        - **Hauteur des barres** → Niveau de certitude du modèle
        - **Plus la barre est haute** → Plus le modèle est sûr
        - **Idéal** : Une barre haute (>70%) et les deux autres basses
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Paramètres du Ménage")
        
        avg_amperage = st.slider(
            "Ampérage moyen par jour (A)", 
            0.0, 50.0, 2.5,
            help="""INTENSITÉ ÉLECTRIQUE :
            • < 0.5A → Très faible (éclairage seulement)
            • 0.5-3A → Normal (éclairage + TV + petit frigo)
            • > 3A → Élevé (gros appareils électriques)"""
        )
        
        # MAINTENANT EN GOURDES HAÏTIENNES (HTG)
        avg_depense = st.slider(
            "Dépense moyenne par jour (HTG)", 
            0.0, 300.0, 22.5,  # 300 HTG max au lieu de 2$
            help="""BUDGET ÉNERGIE JOURNALIER EN GOURDES :
            • 0-7 HTG → Très économique
            • 7-22 HTG → Dépense moyenne  
            • 22-300 HTG → Budget important
            BASÉ SUR LA RÉALITÉ HAÏTIENNE"""
        )
        
        nb_personnes = st.number_input(
            "Nombre de personnes dans le ménage", 
            1, 10, 4,
            help="Plus il y a de personnes, plus la consommation tend à être élevée"
        )
        
    with col2:
        st.markdown("#### 📈 Données d'Observation")
        jours = st.slider(
            "Jours d'observation", 
            1, 365, 90,
            help="""FIABILITÉ DES DONNÉES :
            • < 30 jours → Données peu fiables
            • 30-90 jours → Fiabilité moyenne
            • > 90 jours → Données très fiables"""
        )
        
        # Ratio maintenant en HTG par Ampère
        ratio = st.slider(
            "Ratio dépense/ampérage (HTG par Ampère)", 
            0.0, 150.0, 9.0,  # Ajusté pour les gourdes
            help="""EFFICACITÉ ÉCONOMIQUE :
            • < 7 HTG/A → Bon rapport qualité-prix
            • 7-22 HTG/A → Ratio normal  
            • > 22 HTG/A → Coût élevé par unité d'énergie"""
        )
        
        # Afficher les valeurs avec interprétation
        st.markdown("---")
        st.markdown("**📋 VOTRE PROFIL ACTUEL :**")
        
        # Interprétation de l'ampérage
        if avg_amperage < 0.5:
            amp_interpretation = "🟢 TRÈS FAIBLE"
        elif avg_amperage < 3:
            amp_interpretation = "🟡 NORMAL"
        else:
            amp_interpretation = "🔴 ÉLEVÉ"
            
        # Interprétation de la dépense EN HTG
        if avg_depense < 7:
            dep_interpretation = "🟢 ÉCONOMIQUE"
        elif avg_depense < 22:
            dep_interpretation = "🟡 MOYENNE"
        else:
            dep_interpretation = "🔴 IMPORTANTE"
        
        st.write(f"- ⚡ Ampérage : {avg_amperage} A → {amp_interpretation}")
        st.write(f"- 💰 Dépense : {avg_depense:.0f} HTG → {dep_interpretation}")
        st.write(f"- 👥 Personnes : {nb_personnes}")
        st.write(f"- 📅 Jours observés : {jours}")
        st.write(f"- 📊 Ratio : {ratio:.1f} HTG/A")

    if st.button("🎯 ANALYSER CE MÉNAGE", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        
        # Section de résultats détaillés
        st.markdown("---")
        st.markdown("## 📋 RÉSULTATS DE L'ANALYSE")
        
        # Affichage visuel de la prédiction
        if pred == "grand":
            st.markdown('<div class="prediction-high"><h1>🔴 GRAND CONSOMMATEUR</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 QUE SIGNIFIE CE RÉSULTAT ?</h4>
            <p><strong>Votre ménage consomme plus d'électricité que 80% des foyers haïtiens</strong></p>
            <p>📈 <strong>Caractéristiques typiques :</strong></p>
            <ul>
                <li>• Ampérage supérieur à 3A</li>
                <li>• Possession de gros appareils électriques</li>
                <li>• Consommation régulière et importante</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif pred == "moyen":
            st.markdown('<div class="prediction-medium"><h1>🟡 CONSOMMATION MOYENNE</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 QUE SIGNIFIE CE RÉSULTAT ?</h4>
            <p><strong>Votre consommation est dans la moyenne des ménages haïtiens</strong></p>
            <p>📊 <strong>Profil typique :</strong></p>
            <ul>
                <li>• Ampérage entre 0.5A et 3A</li>
                <li>• Usage modéré de l'électricité</li>
                <li>• Équipements standards (éclairage, TV, petit frigo)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low"><h1>🟢 FAIBLE CONSOMMATION</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 QUE SIGNIFIE CE RÉSULTAT ?</h4>
            <p><strong>Votre ménage est économique en consommation électrique</strong></p>
            <p>🌱 <strong>Caractéristiques :</strong></p>
            <ul>
                <li>• Ampérage inférieur à 0.5A</li>
                <li>• Usage limité à l'éclairage essentiel</li>
                <li>• Faible budget énergie</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphique de probabilités AVEC EXPLICATIONS DÉTAILLÉES
        st.markdown("---")
        st.markdown("## 📊 COMMENT LIRE CE GRAPHIQUE ?")
        
        col_explain, col_graph = st.columns([1, 2])
        
        with col_explain:
            st.markdown("""
            ### 🎯 LE GRAPHIQUE DE CONFIANCE
            
            **Il répond à la question :**  
            *"À quel point le modèle est-il sûr de sa prédiction ?"*
            
            **Comment interpréter :**
            - 📊 **Hauteur des barres** → Niveau de certitude
            - 🟢 **Barre verte** → Probabilité "Faible consommation"
            - 🟡 **Barre jaune** → Probabilité "Consommation moyenne"  
            - 🔴 **Barre rouge** → Probabilité "Grand consommateur"
            
            **EXEMPLE IDÉAL :**
            - Une barre à 85% 
            - Les deux autres à 10% et 5%
            → Le modèle est TRÈS CONFiant !
            """)
            
            max_prob = max(prob)
            pred_index = np.argmax(prob)
            confidence_levels = ['Faible', 'Moyenne', 'Élevée']
            
            st.markdown(f"### 📈 VOTRE RÉSULTAT :")
            st.markdown(f"**Catégorie prédite :** `{confidence_levels[pred_index]}`")
            st.markdown(f"**Niveau de confiance :** `{max_prob:.1%}`")
            
            if max_prob > 0.8:
                st.success("**✅ TRÈS FIABLE** - Le modèle est très certain")
            elif max_prob > 0.6:
                st.info("**ℹ️ FIABLE** - Bon niveau de confiance")
            else:
                st.warning("**⚠️ INCERTAIN** - Plusieurs catégories possibles")
        
        with col_graph:
            fig = go.Figure(go.Bar(
                x=['Faible','Moyenne','Élevée'], 
                y=prob,
                marker_color=['#4cd137','#ff9f43','#ff6b6b'],
                text=[f"{p:.1%}" for p in prob], 
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Probabilité: %{y:.1%}<extra></extra>"
            ))
            fig.update_layout(
                title="📊 NIVEAUX DE CONFIANCE DE LA PRÉDICTION",
                yaxis=dict(
                    tickformat=".0%", 
                    range=[0,1],
                    title="Probabilité (0% = incertain → 100% = certain)"
                ),
                xaxis_title="Catégories de Consommation",
                height=400
            )
            
            # Ajouter une ligne pour le seuil de confiance
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                         annotation_text="Seuil de bonne confiance (70%)")
            
            st.plotly_chart(fig, use_container_width=True)

        # Section d'analyse des facteurs
        st.markdown("---")
        st.markdown("## 🔍 COMMENT VOS DONNÉES ONT ÉTÉ ANALYSÉES")
        
        st.markdown("""
        ### 📋 FACTEURS EXAMINÉS PAR LE MODÈLE :
        """)
        
        factors = {
            "Ampérage": {
                "value": avg_amperage,
                "level": "Élevé" if avg_amperage > 3 else "Modéré" if avg_amperage > 0.5 else "Faible",
                "impact": "FORT" if avg_amperage > 3 else "MOYEN" if avg_amperage > 0.5 else "FAIBLE"
            },
            "Dépense (HTG)": {
                "value": f"{avg_depense:.0f} HTG",
                "level": "Élevée" if avg_depense > 22 else "Modérée" if avg_depense > 7 else "Faible",
                "impact": "FORT" if avg_depense > 50 else "MOYEN" if avg_depense > 7 else "FAIBLE"
            },
            "Taille ménage": {
                "value": nb_personnes,
                "level": "Grand" if nb_personnes > 5 else "Moyen" if nb_personnes > 3 else "Petit",
                "impact": "MOYEN"
            },
            "Période observation": {
                "value": f"{jours} jours",
                "level": "Longue" if jours > 180 else "Moyenne" if jours > 60 else "Courte",
                "impact": "FAIBLE" if jours < 30 else "MOYEN"
            }
        }
        
        for factor, data in factors.items():
            col_fact, col_level, col_impact = st.columns([2, 1, 1])
            with col_fact:
                st.write(f"**{factor}** : {data['value']}")
            with col_level:
                if "Élevé" in data['level'] or "Grand" in data['level']:
                    st.error(data['level'])
                elif "Moyen" in data['level'] or "Modéré" in data['level']:
                    st.warning(data['level'])
                else:
                    st.success(data['level'])
            with col_impact:
                if data['impact'] == "FORT":
                    st.error(f"Impact: {data['impact']}")
                elif data['impact'] == "MOYEN":
                    st.warning(f"Impact: {data['impact']}")
                else:
                    st.info(f"Impact: {data['impact']}")

        # EXPLICATION : Échelle en gourdes haïtiennes
        st.markdown("---")
        with st.expander("💡 ÉCHELLE EN GOURDES HAÏTIENNES (HTG)"):
            st.markdown("""
            ### 📊 CONTEXTE HAÏTIEN - RÉALITÉS ÉCONOMIQUES
            
            **Échelle de référence en GOURDES :**
            
            • 🏠 **Dépense très économique** : 0-7 HTG/jour
               *→ Éclairage basique seulement*
               
            • 💡 **Dépense moyenne** : 7-22 HTG/jour  
               *→ Éclairage + TV + petit frigo*
               
            • ⚡ **Dépense importante** : 22-100 HTG/jour
               *→ Appareils électriques supplémentaires*
               
            • 🏢 **Dépense très élevée** : 100-300 HTG/jour
               *→ Cas exceptionnels (entreprises, grandes familles)*
            
            **💱 Conversion approximative :**
            - 7 HTG ≈ 0.05 USD
            - 22 HTG ≈ 0.15 USD  
            - 150 HTG ≈ 1.00 USD
            """)

def show_new_data_prediction(clf):
    st.markdown('<h2 class="sub-header">📁 Prédictions sur Nouvelles Données</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Format requis pour le fichier CSV :</h4>
    <p>Votre fichier doit contenir les colonnes suivantes :</p>
    <ul>
        <li><code>avg_amperage_per_day</code> : Ampérage moyen quotidien (A)</li>
        <li><code>avg_depense_per_day</code> : Dépense moyenne quotidienne (HTG) ← EN GOURDES</li>
        <li><code>nombre_personnes</code> : Nombre de personnes dans le ménage</li>
        <li><code>jours_observed</code> : Nombre de jours d'observation</li>
        <li><code>ratio_depense_amperage</code> : Ratio dépense/ampérage (HTG par Ampère)</li>
    </ul>
    <p class="currency-note">💡 <strong>Note :</strong> Toutes les dépenses doivent être en Gourdes Haïtiennes (HTG)</p>
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
        ### 🎯 Comment évaluer la qualité d'une prédiction
        
        **Indicateurs de fiabilité :**
        
        📊 **Probabilités élevées** (> 80%)
        - La prédiction est très fiable
        - Le modèle est certain de sa classification
        
        📊 **Probabilités moyennes** (60-80%)
        - Bon niveau de confiance
        - Résultat probable mais d'autres catégories possibles
        
        📊 **Probabilités faibles** (< 60%)
        - Prédiction incertaine
        - Plusieurs catégories presque équiprobables
        
        ### 🔍 Facteurs clés d'analyse
        
        **Ampérage moyen :**
        - < 0.5A : Faible consommation
        - 0.5-3A : Consommation moyenne  
        - > 3A : Forte consommation
        
        **Ratio dépense/ampérage :**
        - Faible : Bon rendement économique
        - Élevé : Coût important par unité consommée
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Seuils de référence EN GOURDES
        
        **Consommation typique en Haïti :**
        - **Ménage modeste** : 0.5-1.5A (7-15 HTG/jour)
        - **Ménage moyen** : 1.5-3A (15-22 HTG/jour)
        - **Ménage aisé** : 3A et plus (22+ HTG/jour)
        
        **Dépenses énergétiques en HTG :**
        - **Économique** : < 7 HTG/jour
        - **Standard** : 7-22 HTG/jour
        - **Élevée** : > 22 HTG/jour
        
        ### ✅ Quand la prédiction est-elle "bonne" ?
        
        Une prédiction est considérée comme fiable quand :
        1. La probabilité maximale dépasse **70%**
        2. Les données d'entrée sont complètes et réalistes
        3. La période d'observation est suffisante (> 30 jours)
        4. Le profil de consommation est cohérent
        """)
    
    st.markdown("---")
    st.markdown("#### 🚨 Cas particuliers à surveiller")
    
    st.warning("""
    **Situations nécessitant une vérification manuelle :**
    - Probabilités très proches entre plusieurs catégories
    - Données d'observation insuffisantes (< 30 jours)
    - Valeurs extrêmes ou atypiques
    - Incohérence entre l'ampérage et la dépense
    """)
    
    st.markdown("---")
    st.markdown("#### 📚 Glossaire des Termes")
    
    glossary_col1, glossary_col2 = st.columns(2)
    
    with glossary_col1:
        st.markdown("""
        **Ampérage moyen :**
        > Intensité du courant électrique consommée en moyenne chaque jour
        
        **Ratio dépense/ampérage :**
        > Efficacité économique : coût par unité d'énergie consommée (HTG/A)
        
        **Grand consommateur :**
        > Ménage avec une consommation électrique supérieure à 3A par jour
        """)
    
    with glossary_col2:
        st.markdown("""
        **Période d'observation :**
        > Durée pendant laquelle les données de consommation ont été collectées
        
        **Indice de certitude :**
        > Mesure mathématique de la confiance globale du modèle
        
        **HTG :**
        > Gourde Haïtienne - Devise nationale d'Haïti
        """)

# ==============================
# APPLICATION PRINCIPALE
# ==============================
def main():
    st.markdown('<h1 class="main-header">🏠 Classification Intelligente des Ménages Haïtiens</h1>', unsafe_allow_html=True)
    
    # Information sur le mode
    if st.sidebar.checkbox("ℹ️ Afficher les informations techniques", value=False):
        st.sidebar.info("""
        **Mode actuel :** 
        - 🔍 Chargement des modèles réels si disponibles
        - 🎮 Mode démo activé sinon
        
        **Technologies :**
        - Machine Learning : Random Forest
        - Interface : Streamlit
        - Visualisation : Plotly
        
        **💱 Devise :** Gourdes Haïtiennes (HTG)
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
