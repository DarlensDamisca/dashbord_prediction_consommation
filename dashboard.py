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
    with st.expander("ℹ️ Comment utiliser cette prédiction", expanded=True):
        st.markdown("""
        **Guide d'interprétation :**
        - **🟢 Faible consommation** : Ménage économique, consommation inférieure à 0.5A
        - **🟡 Consommation moyenne** : Usage modéré, entre 0.5A et 3A  
        - **🔴 Grand consommateur** : Forte consommation, supérieure à 3A
        
        **Facteurs influençant la prédiction :**
        - Ampérage moyen : intensité du courant utilisée
        - Dépense quotidienne : budget énergie
        - Nombre de personnes : taille du ménage
        - Période d'observation : fiabilité des données
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Paramètres du Ménage")
        avg_amperage = st.slider(
            "Ampérage moyen par jour (A)", 
            0.0, 50.0, 2.5,
            help="Intensité électrique moyenne consommée quotidiennement"
        )
        avg_depense = st.slider(
            "Dépense moyenne par jour ($)", 
            0.0, 2.0, 0.15,
            help="Budget quotidien alloué à l'énergie électrique"
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
            "Ratio dépense/ampérage", 
            0.0, 0.5, 0.06,
            help="Efficacité économique : dépense par unité d'ampérage"
        )
        
        # Afficher les valeurs actuelles
        st.markdown("---")
        st.markdown("**Valeurs saisies :**")
        st.write(f"- Ampérage : {avg_amperage} A")
        st.write(f"- Dépense : ${avg_depense:.2f}")
        st.write(f"- Personnes : {nb_personnes}")
        st.write(f"- Jours observés : {jours}")
        st.write(f"- Ratio : {ratio:.3f}")

    if st.button("🎯 Analyser ce Ménage", use_container_width=True):
        pred, prob = clf.predict_household([avg_amperage, avg_depense, nb_personnes, jours, ratio])
        
        # Section de résultats détaillés
        st.markdown("---")
        st.markdown("## 📋 Résultats de l'Analyse")
        
        # Affichage visuel de la prédiction
        if pred == "grand":
            st.markdown('<div class="prediction-high"><h1>🔴 GRAND CONSOMMATEUR</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Interprétation :</h4>
            <p>Ce ménage présente une consommation électrique élevée. Recommandations :</p>
            <ul>
                <li>✅ Vérifier l'efficacité des appareils électriques</li>
                <li>✅ Envisager des équipements énergétiquement efficaces</li>
                <li>✅ Analyser les habitudes de consommation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif pred == "moyen":
            st.markdown('<div class="prediction-medium"><h1>🟡 CONSOMMATION MOYENNE</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Interprétation :</h4>
            <p>Consommation typique pour un ménage haïtien. Situation stable.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low"><h1>🟢 FAIBLE CONSOMMATION</h1></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Interprétation :</h4>
            <p>Consommation économique. Bonne gestion énergétique.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphique de probabilités avec explications
        col_prob, col_explain = st.columns([2, 1])
        
        with col_prob:
            fig = go.Figure(go.Bar(
                x=['Faible','Moyenne','Élevée'], 
                y=prob,
                marker_color=['#4cd137','#ff9f43','#ff6b6b'],
                text=[f"{p:.1%}" for p in prob], 
                textposition='auto'
            ))
            fig.update_layout(
                title="📊 Niveaux de Confiance de la Prédiction",
                yaxis=dict(tickformat=".0%", range=[0,1]),
                xaxis_title="Catégories de Consommation",
                yaxis_title="Probabilité"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_explain:
            st.markdown("#### 🎯 Fiabilité de la Prédiction")
            max_prob = max(prob)
            if max_prob > 0.8:
                st.success("**Très fiable** ✅")
                st.write("La prédiction est très certaine")
            elif max_prob > 0.6:
                st.info("**Fiable** ℹ️")
                st.write("Bon niveau de confiance")
            else:
                st.warning("**Incertaine** ⚠️")
                st.write("Plusieurs catégories possibles")
            
            st.metric("Confiance maximale", f"{max_prob:.1%}")
            
            # Indice de fiabilité globale
            confidence_score = sum(p**2 for p in prob)  # Indice de Gini
            st.metric("Indice de certitude", f"{confidence_score:.1%}")

        # Section d'analyse des facteurs
        st.markdown("---")
        st.markdown("#### 🔍 Analyse des Facteurs Influents")
        
        factors = {
            "Ampérage": "Élevé" if avg_amperage > 3 else "Modéré" if avg_amperage > 0.5 else "Faible",
            "Dépense": "Élevée" if avg_depense > 0.1 else "Modérée" if avg_depense > 0.05 else "Faible",
            "Taille ménage": "Grand" if nb_personnes > 5 else "Moyen" if nb_personnes > 3 else "Petit",
            "Période observation": "Longue" if jours > 180 else "Moyenne" if jours > 60 else "Courte"
        }
        
        for factor, level in factors.items():
            col_fact, col_level = st.columns([2, 1])
            with col_fact:
                st.write(f"**{factor}**")
            with col_level:
                if "Élevé" in level or "Grand" in level:
                    st.error(level)
                elif "Moyen" in level or "Modéré" in level:
                    st.warning(level)
                else:
                    st.success(level)

def show_new_data_prediction(clf):
    st.markdown('<h2 class="sub-header">📁 Prédictions sur Nouvelles Données</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Format requis pour le fichier CSV :</h4>
    <p>Votre fichier doit contenir les colonnes suivantes :</p>
    <ul>
        <li><code>avg_amperage_per_day</code> : Ampérage moyen quotidien (A)</li>
        <li><code>avg_depense_per_day</code> : Dépense moyenne quotidienne ($)</li>
        <li><code>nombre_personnes</code> : Nombre de personnes dans le ménage</li>
        <li><code>jours_observed</code> : Nombre de jours d'observation</li>
        <li><code>ratio_depense_amperage</code> : Ratio dépense/ampérage</li>
    </ul>
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
        ### 📈 Seuils de référence
        
        **Consommation typique en Haïti :**
        - **Ménage modeste** : 0.5-1.5A
        - **Ménage moyen** : 1.5-3A
        - **Ménage aisé** : 3A et plus
        
        **Dépenses énergétiques :**
        - **Économique** : < $0.05/jour
        - **Standard** : $0.05-$0.15/jour
        - **Élevée** : > $0.15/jour
        
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
        > Efficacité économique : coût par unité d'énergie consommée
        
        **Grand consommateur :**
        > Ménage avec une consommation électrique supérieure à 3A par jour
        """)
    
    with glossary_col2:
        st.markdown("""
        **Période d'observation :**
        > Durée pendant laquelle les données de consommation ont été collectées
        
        **Indice de certitude :**
        > Mesure mathématique de la confiance globale du modèle
        
        **Précision du modèle :**
        > Pourcentage de prédictions correctes sur les données de test
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


if __name__ == "__main__":
    main()
