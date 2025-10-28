import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

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
    .appliance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
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

    def predict_batch(self, df):
        """Faire des prédictions sur un lot de données"""
        try:
            # Vérifier les colonnes requises
            required_columns = ['avg_amperage_per_day', 'avg_depense_per_day', 
                              'nombre_personnes', 'jours_observed']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None, f"Colonnes manquantes: {missing_columns}"
            
            # Calculer le ratio si absent
            if 'ratio_depense_amperage' not in df.columns:
                df['ratio_depense_amperage'] = df['avg_depense_per_day'] / df['avg_amperage_per_day']
                df['ratio_depense_amperage'] = df['ratio_depense_amperage'].replace([np.inf, -np.inf], 0)
                df['ratio_depense_amperage'] = df['ratio_depense_amperage'].fillna(0)
            
            # Préparer les données
            X = df[self.features].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Prédictions
            predictions_encoded = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Décoder les prédictions
            predictions_decoded = self.label_encoder.inverse_transform(predictions_encoded)
            
            # Créer le DataFrame de résultats
            results_df = df.copy()
            results_df['niveau_conso_pred'] = predictions_decoded
            results_df['prob_petit'] = probabilities[:, 0]
            results_df['prob_moyen'] = probabilities[:, 1]
            results_df['prob_grand'] = probabilities[:, 2]
            results_df['confiance'] = np.max(probabilities, axis=1)
            
            return results_df, None
            
        except Exception as e:
            return None, f"Erreur lors de la prédiction par lot: {e}"

class ApplianceCalculator:
    def __init__(self):
        # Base de données des appareils électriques typiques en Haïti (en Watts)
        self.appliance_db = {
            # Éclairage
            'Ampoule LED 10W': {'power_w': 10, 'usage_hours': 6, 'category': 'Éclairage'},
            'Ampoule LED 15W': {'power_w': 15, 'usage_hours': 6, 'category': 'Éclairage'},
            'Ampoule Fluorescente 20W': {'power_w': 20, 'usage_hours': 6, 'category': 'Éclairage'},
            'Tube Fluorescent 40W': {'power_w': 40, 'usage_hours': 8, 'category': 'Éclairage'},
            
            # Électronique
            'Téléphone Portable (Charge)': {'power_w': 10, 'usage_hours': 4, 'category': 'Électronique'},
            'Laptop': {'power_w': 60, 'usage_hours': 6, 'category': 'Électronique'},
            'Desktop PC': {'power_w': 200, 'usage_hours': 4, 'category': 'Électronique'},
            'TV LED 32"': {'power_w': 50, 'usage_hours': 5, 'category': 'Électronique'},
            'TV LCD 42"': {'power_w': 120, 'usage_hours': 5, 'category': 'Électronique'},
            'Radio': {'power_w': 15, 'usage_hours': 8, 'category': 'Électronique'},
            
            # Électroménager
            'Réfrigérateur (Classe A)': {'power_w': 150, 'usage_hours': 8, 'category': 'Électroménager'},
            'Réfrigérateur (Vieux Modèle)': {'power_w': 300, 'usage_hours': 12, 'category': 'Électroménager'},
            'Ventilateur de Plafond': {'power_w': 75, 'usage_hours': 8, 'category': 'Électroménager'},
            'Ventilateur sur Pied': {'power_w': 50, 'usage_hours': 6, 'category': 'Électroménager'},
            'Blender/Mixeur': {'power_w': 300, 'usage_hours': 0.5, 'category': 'Électroménager'},
            'Fer à Repasser': {'power_w': 1000, 'usage_hours': 1, 'category': 'Électroménager'},
            'Machine à Laver': {'power_w': 500, 'usage_hours': 1, 'category': 'Électroménager'},
            'Climatiseur 9000 BTU': {'power_w': 900, 'usage_hours': 4, 'category': 'Électroménager'},
            'Climatiseur 12000 BTU': {'power_w': 1200, 'usage_hours': 4, 'category': 'Électroménager'},
            
            # Cuisine
            'Plaque de Cuisson Électrique': {'power_w': 1500, 'usage_hours': 2, 'category': 'Cuisine'},
            'Four Micro-ondes': {'power_w': 800, 'usage_hours': 0.5, 'category': 'Cuisine'},
            'Bouilloire Électrique': {'power_w': 1500, 'usage_hours': 0.3, 'category': 'Cuisine'},
            'Réchaud Électrique': {'power_w': 1000, 'usage_hours': 1, 'category': 'Cuisine'},
            
            # Énergie
            'Backup Stockage Énergie': {'power_w': 50, 'usage_hours': 24, 'category': 'Énergie'},
            'Onduleur (UPS)': {'power_w': 100, 'usage_hours': 24, 'category': 'Énergie'},
            'Chargeur Solaire': {'power_w': 20, 'usage_hours': 6, 'category': 'Énergie'},
            
            # Divers
            'Pompe à Eau': {'power_w': 500, 'usage_hours': 1, 'category': 'Divers'},
            'Sèche-Cheveux': {'power_w': 1200, 'usage_hours': 0.3, 'category': 'Divers'},
            'Aspirateur': {'power_w': 800, 'usage_hours': 0.5, 'category': 'Divers'}
        }
    
    def calculate_consumption(self, selected_appliances):
        """Calculer la consommation totale basée sur les appareils sélectionnés"""
        total_energy_wh = 0
        consumption_by_category = {}
        
        for appliance, quantity in selected_appliances.items():
            if quantity > 0 and appliance in self.appliance_db:
                appliance_data = self.appliance_db[appliance]
                daily_energy = appliance_data['power_w'] * appliance_data['usage_hours'] * quantity
                total_energy_wh += daily_energy
                
                category = appliance_data['category']
                if category not in consumption_by_category:
                    consumption_by_category[category] = 0
                consumption_by_category[category] += daily_energy
        
        # Convertir en kWh et estimer l'ampérage (supposant 120V)
        total_energy_kwh = total_energy_wh / 1000
        estimated_amperage = (total_energy_kwh * 1000) / (120 * 24)  # I = P/V
        
        # Estimation des dépenses (environ $0.25/kWh en Haïti)
        estimated_cost = total_energy_kwh * 0.25
        
        return {
            'total_energy_kwh': total_energy_kwh,
            'estimated_amperage': estimated_amperage,
            'estimated_cost': estimated_cost,
            'consumption_by_category': consumption_by_category,
            'total_energy_wh': total_energy_wh
        }

def display_consumption_calculation(consumption_data, selected_appliances, appliance_calc):
    """Afficher les résultats du calcul de consommation"""
    
    st.header("📊 Résultats du Calcul de Consommation")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Consommation Quotidienne", 
            f"{consumption_data['total_energy_kwh']:.2f} kWh"
        )
    
    with col2:
        st.metric(
            "Ampérage Estimé", 
            f"{consumption_data['estimated_amperage']:.2f} A"
        )
    
    with col3:
        st.metric(
            "Coût Quotidien Estimé", 
            f"${consumption_data['estimated_cost']:.2f}"
        )
    
    with col4:
        st.metric(
            "Énergie Totale", 
            f"{consumption_data['total_energy_wh']:.0f} Wh"
        )
    
    # Graphiques
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Consommation par catégorie
        if consumption_data['consumption_by_category']:
            fig_pie = px.pie(
                values=list(consumption_data['consumption_by_category'].values()),
                names=list(consumption_data['consumption_by_category'].keys()),
                title="Répartition de la Consommation par Catégorie",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        # Appareils les plus consommateurs
        appliance_consumption = []
        for appliance, quantity in selected_appliances.items():
            if quantity > 0 and appliance in appliance_calc.appliance_db:
                data = appliance_calc.appliance_db[appliance]
                consumption = data['power_w'] * data['usage_hours'] * quantity
                appliance_consumption.append({
                    'Appareil': appliance,
                    'Consommation (Wh)': consumption
                })
        
        if appliance_consumption:
            df_consumption = pd.DataFrame(appliance_consumption)
            df_consumption = df_consumption.sort_values('Consommation (Wh)', ascending=True)
            
            fig_bar = px.bar(
                df_consumption,
                y='Appareil',
                x='Consommation (Wh)',
                title="Consommation par Appareil",
                orientation='h',
                color='Consommation (Wh)',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Détails des calculs
    with st.expander("📋 Détails des Calculs"):
        st.subheader("Calculs Détaillés par Appareil")
        
        calculation_details = []
        for appliance, quantity in selected_appliances.items():
            if quantity > 0 and appliance in appliance_calc.appliance_db:
                data = appliance_calc.appliance_db[appliance]
                daily_wh = data['power_w'] * data['usage_hours'] * quantity
                calculation_details.append({
                    'Appareil': appliance,
                    'Quantité': quantity,
                    'Puissance (W)': data['power_w'],
                    'Heures/jour': data['usage_hours'],
                    'Consommation (Wh/jour)': daily_wh,
                    'Catégorie': data['category']
                })
        
        if calculation_details:
            df_details = pd.DataFrame(calculation_details)
            st.dataframe(df_details)

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
        st.metric("Ampérage Quotidien", f"{input_data['avg_amperage_per_day']:.2f} A")
        st.metric("Dépenses Quotidiennes", f"${input_data['avg_depense_per_day']:.2f}")
    
    # Recommandations basées sur la prédiction
    st.subheader("🎯 Recommandations Personnalisées")
    
    recommendations = {
        'petit': [
            "💡 Continuez vos bonnes habitudes de consommation"   
        ],
        'moyen': [
            "⚠️ **Consommation moyenne** - Potentiel d'optimisation",
            "🔍 Identifiez les appareils les plus énergivores",
            "📈 Surveillez les pics de consommation",
            "💡 Remplacez les vieux appareils par des modèles efficaces",
            "⏰ Utilisez les appareils en dehors des heures de pointe"
        ],
        'grand': [
            "🚨 **Forte consommation détectée** - Action recommandée",
            "🔧 **Audit énergétique urgent** nécessaire",
            "💡 Remplacez immédiatement les appareils énergivores",
            "🌡️ Réduisez l'usage du climatiseur lorsque possible",
            "⚡ Envisagez des solutions énergétiques alternatives",
            "📋 Établissez un plan de réduction de consommation"
        ]
    }
    
    for rec in recommendations.get(prediction, []):
        st.write(rec)

def show_appliance_prediction(predictor, appliance_calc):
    """Interface pour la prédiction basée sur les appareils"""
    
    st.header("🏠 Prédiction par Appareils Électroménagers")
    
    st.markdown("""
    ### 📋 Instructions
    Sélectionnez les appareils électriques utilisés dans votre ménage et leur quantité.
    Le système calculera automatiquement la consommation estimée et prédira le niveau de consommation.
    """)
    
    # Organisation des appareils par catégorie
    categories = {}
    for appliance, data in appliance_calc.appliance_db.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(appliance)
    
    # Interface de sélection des appareils
    selected_appliances = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛋️ Éclairage et Électronique")
        
        for category in ['Éclairage', 'Électronique']:
            if category in categories:
                st.markdown(f"**{category}**")
                for appliance in categories[category]:
                    quantity = st.number_input(
                        f"{appliance}",
                        min_value=0,
                        max_value=10,
                        value=0,
                        key=f"app_{appliance}"
                    )
                    selected_appliances[appliance] = quantity
                st.markdown("---")
    
    with col2:
        st.subheader("🍳 Électroménager et Cuisine")
        
        for category in ['Électroménager', 'Cuisine', 'Énergie', 'Divers']:
            if category in categories:
                st.markdown(f"**{category}**")
                for appliance in categories[category]:
                    quantity = st.number_input(
                        f"{appliance}",
                        min_value=0,
                        max_value=10,
                        value=0,
                        key=f"app_{appliance}"
                    )
                    selected_appliances[appliance] = quantity
                st.markdown("---")
    
    # Informations supplémentaires
    st.subheader("📊 Informations du Ménage")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        nombre_personnes = st.number_input(
            "Nombre de personnes dans le foyer",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            key="app_nb_pers"
        )
    
    with col_info2:
        jours_observed = st.number_input(
            "Nombre de jours d'observation",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            key="app_jours"
        )
    
    with col_info3:
        zone = st.selectbox(
            "Zone géographique",
            ["Zone Inconnue", "Môle Saint-Nicolas", "Jean Rabel", "Bombardopolis", "Mare-Rouge"],
            key="app_zone"
        )
    
    # Bouton de calcul et prédiction
    if st.button("⚡ Calculer et Prédire la Consommation", type="primary"):
        with st.spinner("Calcul de la consommation..."):
            # Calcul de la consommation
            consumption_data = appliance_calc.calculate_consumption(selected_appliances)
            
            # Affichage des résultats du calcul
            display_consumption_calculation(consumption_data, selected_appliances, appliance_calc)
            
            # Préparation pour la prédiction
            input_data = {
                'avg_amperage_per_day': consumption_data['estimated_amperage'],
                'avg_depense_per_day': consumption_data['estimated_cost'],
                'nombre_personnes': nombre_personnes,
                'jours_observed': jours_observed,
                'ratio_depense_amperage': consumption_data['estimated_cost'] / consumption_data['estimated_amperage'] if consumption_data['estimated_amperage'] > 0 else 0
            }
            
            # Prédiction
            result = predictor.predict(input_data)
            
            if result:
                display_prediction_result(result, input_data)

def show_single_prediction(predictor):
    """Interface pour la prédiction simple"""
    
    st.header("🔮 Prédiction Simple de Consommation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres du Ménage")
        
        avg_amperage = st.number_input(
            "Ampérage moyen quotidien (A)",
            min_value=0.0,
            max_value=100.0,
            value=1.5,
            step=0.1
        )
        
        avg_depense = st.number_input(
            "Dépenses moyennes quotidiennes ($)",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.01
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
            step=1
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
        
        if avg_amperage > 0:
            ratio = avg_depense / avg_amperage
        else:
            ratio = 0
        
        st.metric("Ratio Dépenses/Ampérage", f"{ratio:.4f}")
        
        if st.button("🔍 Prédire le Niveau de Consommation", type="primary"):
            input_data = {
                'avg_amperage_per_day': avg_amperage,
                'avg_depense_per_day': avg_depense,
                'nombre_personnes': nombre_personnes,
                'jours_observed': jours_observed,
                'ratio_depense_amperage': ratio
            }
            
            result = predictor.predict(input_data)
            
            if result:
                display_prediction_result(result, input_data)

def show_batch_prediction(predictor):
    """Interface pour les prédictions par lot"""
    
    st.header("📊 Prédiction par Lot")
    
    st.markdown("""
    ### 📋 Instructions
    Téléchargez un fichier CSV contenant les données de plusieurs ménages. 
    Le fichier doit contenir au minimum les colonnes suivantes:
    
    **Colonnes requises:**
    - `avg_amperage_per_day` : Ampérage moyen quotidien
    - `avg_depense_per_day` : Dépenses moyennes quotidiennes ($)
    - `nombre_personnes` : Nombre de personnes dans le foyer
    - `jours_observed` : Nombre de jours d'observation
    
    **Colonnes optionnelles:**
    - `ratio_depense_amperage` : Ratio dépenses/ampérage (calculé automatiquement si absent)
    - Autres colonnes d'identification (numero_compteur, nom_complet, etc.)
    """)
    
    # Template de fichier CSV
    st.subheader("📁 Template de Fichier CSV")
    
    #  template exemple
    template_data = {
    'numero_compteur': ['#001', '#002', '#003', '#004', '#005', '#006', '#007', '#008', '#009', '#010', '#011', '#012', '#013', '#014', '#015', '#016', '#017', '#018', '#019', '#020', '#021', '#022', '#023'],
    'nom_complet': ['Jean Dupont', 'Marie Laurent', 'Pierre Martin', 'Sophie Alexandre', 'Marc Antoine', 'Nadia Joseph', 'Robert Desir', 'Isabelle Moreau', 'Daniel Thomas', 'Caroline Baptiste', 'Patrick Noel', 'Vanessa Pierre', 'Samuel Jean', 'Christelle Laurent', 'Michel Olivier', 'Nicole St-Pierre', 'Emmanuel Toussaint', 'Karen Benjamin', 'Wilson Charles', 'Sandra Fleury', 'Georges Laguerre', 'Mireille Dorval', 'Fritznel Mentor'],
    'avg_amperage_per_day': [0.5, 2.0, 5.0, 1.2, 3.5, 0.8, 4.2, 1.5, 2.8, 0.6, 6.0, 1.0, 3.0, 2.2, 4.5, 0.9, 5.5, 1.8, 2.5, 3.8, 1.2, 4.0, 0.7],
    'avg_depense_per_day': [0.1, 0.5, 1.2, 0.3, 0.8, 0.2, 1.0, 0.4, 0.7, 0.15, 1.5, 0.25, 0.75, 0.55, 1.1, 0.22, 1.3, 0.45, 0.62, 0.95, 0.3, 1.0, 0.18],
    'nombre_personnes': [3, 4, 6, 2, 5, 3, 7, 4, 5, 2, 8, 3, 4, 5, 6, 3, 7, 4, 5, 4, 3, 6, 2],
    'jours_observed': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    'zone': ['Port-au-Prince', 'Cap-Haïtien', 'Gonaïves', 'Port-au-Prince', 'Jacmel', 'Les Cayes', 'Cap-Haïtien', 'Port-au-Prince', 'Gonaïves', 'Jérémie', 'Port-au-Prince', 'Cap-Haïtien', 'Les Cayes', 'Jacmel', 'Port-au-Prince', 'Gonaïves', 'Cap-Haïtien', 'Port-au-Prince', 'Les Cayes', 'Jacmel', 'Jérémie', 'Port-au-Prince', 'Gonaïves']
}
    
    template_df = pd.DataFrame(template_data)
    
    col_template1, col_template2 = st.columns(2)
    
    with col_template1:
        st.dataframe(template_df, use_container_width=True)
    
    with col_template2:
        # Télécharger le template
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger le Template CSV",
            data=csv_template,
            file_name="template_menages.csv",
            mime="text/csv",
            help="Téléchargez ce template et remplissez-le avec vos données"
        )
    
    # Upload de fichier
    st.subheader("📤 Upload du Fichier de Données")
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type="csv",
        help="Sélectionnez un fichier CSV contenant les données des ménages"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Fichier chargé avec succès: {len(df)} enregistrements")
            
            # Aperçu des données
            st.subheader("👀 Aperçu des Données Chargées")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Informations sur les données
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.metric("Nombre d'enregistrements", len(df))
                st.metric("Nombre de colonnes", len(df.columns))
            
            with col_info2:
                # Vérifier les colonnes requises
                required_columns = ['avg_amperage_per_day', 'avg_depense_per_day', 
                                  'nombre_personnes', 'jours_observed']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Colonnes manquantes: {missing_columns}")
                else:
                    st.success("✅ Toutes les colonnes requises sont présentes")
            
            # Bouton de prédiction
            if st.button("🚀 Lancer les Prédictions par Lot", type="primary", disabled=len(missing_columns) > 0):
                with st.spinner("Traitement des prédictions en cours..."):
                    # Prédictions par lot
                    results_df, error = predictor.predict_batch(df)
                    
                    if error:
                        st.error(f"Erreur: {error}")
                    else:
                        st.success(f"✅ Prédictions terminées pour {len(results_df)} ménages")
                        
                        # Affichage des résultats
                        st.subheader("📈 Résultats des Prédictions")
                        
                        # Statistiques globales
                        st.markdown("### 📊 Statistiques Globales")
                        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                        
                        with col_stats1:
                            count_petit = (results_df['niveau_conso_pred'] == 'petit').sum()
                            st.metric("Petits Consommateurs", count_petit)
                        
                        with col_stats2:
                            count_moyen = (results_df['niveau_conso_pred'] == 'moyen').sum()
                            st.metric("Moyens Consommateurs", count_moyen)
                        
                        with col_stats3:
                            count_grand = (results_df['niveau_conso_pred'] == 'grand').sum()
                            st.metric("Grands Consommateurs", count_grand)
                        
                        with col_stats4:
                            avg_confidence = results_df['confiance'].mean()
                            st.metric("Confiance Moyenne", f"{avg_confidence*100:.1f}%")
                        
                        # Visualisation de la distribution
                        st.markdown("### 📊 Distribution des Prédictions")
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Camembert de distribution
                            dist_data = results_df['niveau_conso_pred'].value_counts()
                            fig_pie = px.pie(
                                values=dist_data.values,
                                names=dist_data.index,
                                title="Répartition des Niveaux de Consommation",
                                color=dist_data.index,
                                color_discrete_map={
                                    'petit': '#00cc96', 
                                    'moyen': '#ffa500', 
                                    'grand': '#ff4b4b'
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_viz2:
                            # Histogramme de confiance
                            fig_hist = px.histogram(
                                results_df,
                                x='confiance',
                                nbins=20,
                                title="Distribution de la Confiance des Prédictions",
                                color_discrete_sequence=['#1f77b4']
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Tableau des résultats détaillés
                        st.markdown("### 📋 Détails des Prédictions")
                        
                        # Sélection des colonnes à afficher
                        default_columns = ['numero_compteur', 'nom_complet', 'niveau_conso_pred', 
                                         'confiance', 'avg_amperage_per_day', 'avg_depense_per_day']
                        
                        available_columns = [col for col in default_columns if col in results_df.columns]
                        available_columns.extend(['prob_petit', 'prob_moyen', 'prob_grand'])
                        
                        display_df = results_df[available_columns]
                        
                        # Formater les pourcentages
                        if 'confiance' in display_df.columns:
                            display_df['confiance'] = (display_df['confiance'] * 100).round(1).astype(str) + '%'
                        
                        for col in ['prob_petit', 'prob_moyen', 'prob_grand']:
                            if col in display_df.columns:
                                display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Téléchargement des résultats
                        st.markdown("### 💾 Téléchargement des Résultats")
                        
                        # Options de format
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # CSV
                            csv_results = results_df.to_csv(index=False)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                            
                            st.download_button(
                                label="📥 Télécharger en CSV",
                                data=csv_results,
                                file_name=f"predictions_menages_{timestamp}.csv",
                                mime="text/csv",
                                help="Téléchargez tous les résultats au format CSV"
                            )
                        
                        with col_dl2:
                            # Excel
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                
                                # Ajouter un sheet avec les statistiques
                                stats_data = {
                                    'Statistique': ['Total ménages', 'Petits consommateurs', 
                                                   'Moyens consommateurs', 'Grands consommateurs',
                                                   'Confiance moyenne'],
                                    'Valeur': [len(results_df), count_petit, count_moyen, 
                                              count_grand, f"{avg_confidence*100:.1f}%"]
                                }
                                pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistiques', index=False)
                            
                            excel_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Télécharger en Excel",
                                data=excel_buffer,
                                file_name=f"predictions_menages_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Téléchargez tous les résultats au format Excel avec statistiques"
                            )
                        
                        # Résumé exécutif
                        with st.expander("📄 Résumé Exécutif"):
                            st.markdown(f"""
                            ### Résumé des Prédictions
                            
                            - **Total des ménages analysés** : {len(results_df)}
                            - **Petits consommateurs** : {count_petit} ({count_petit/len(results_df)*100:.1f}%)
                            - **Moyens consommateurs** : {count_moyen} ({count_moyen/len(results_df)*100:.1f}%)
                            - **Grands consommateurs** : {count_grand} ({count_grand/len(results_df)*100:.1f}%)
                            - **Confiance moyenne des prédictions** : {avg_confidence*100:.1f}%
                            
                            **Recommandations:**
                            - Segmenter les stratégies énergétiques selon la distribution des consommateurs
                            - Cibler les {count_grand} grands consommateurs pour des programmes d'efficacité énergétique
                            - Maintenir les {count_petit} petits consommateurs avec des tarifs avantageux
                            """)
                        
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {e}")
            st.info("Vérifiez que le fichier est un CSV valide et qu'il contient les colonnes requises.")

def show_analytics():
    """Page d'analytics et de visualisations"""
    st.header("📈 Analytics et Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Performance du Modèle", "99.8%")
    
    with col2:
        st.metric("Précision", "99.8%")
    
    with col3:
        st.metric("Taux d'Erreur", "0.2%")
    
    with col4:
        st.metric("Données d'Entraînement", "2,716 foyers")

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
    
    ### 👥 Équipe
    - Darlens Damisca
    - Saint Germain Emode
    """)

def main():
    # En-tête de l'application
    st.markdown('<h1 class="main-header">⚡ Classification des Ménages Haïtiens</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette application utilise un modèle de machine learning pour classifier les ménages haïtiens 
    selon leur niveau de consommation énergétique (faible, moyen, élevé).
    """)
    
    # Initialisation des classes
    predictor = ConsumptionPredictor()
    appliance_calc = ApplianceCalculator()
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choisir le mode",
        [
            "🔮 Prédiction Simple", 
            "🏠 Prédiction par Appareils", 
            "📊 Batch Prediction", 
            "📈 Analytics", 
            "ℹ️ A propos"
        ]
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
    
    if app_mode == "🔮 Prédiction Simple":
        show_single_prediction(predictor)
    elif app_mode == "🏠 Prédiction par Appareils":
        show_appliance_prediction(predictor, appliance_calc)
    elif app_mode == "📊 Batch Prediction":
        show_batch_prediction(predictor)
    elif app_mode == "📈 Analytics":
        show_analytics()
    else:
        show_about()

if __name__ == "__main__":
    main()
