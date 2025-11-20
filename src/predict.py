import pandas as pd
import joblib
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_best_pipeline.pkl"

STRING_RISK_CLASSES = np.array(['Critical', 'High', 'Low', 'Medium'])

def cargar_pipeline(path):
    """Carga la pipeline completa."""
    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        print(f"Error: Archivo de modelo no encontrado en {path}. Ejecuta train_xgboost.py primero.")
        return None
    except Exception as e:
        print(f"Error al cargar la pipeline: {e}")
        return None

def main_predict():
    # Cargar la pipeline completa
    full_pipeline = cargar_pipeline(MODEL_PATH)
    if full_pipeline is None:
        return

    # Verificar que el número de clases coincida
    try:
        numerical_classes = full_pipeline.named_steps['classifier'].classes_
        if len(numerical_classes) != len(STRING_RISK_CLASSES):
            print(f"ERROR DE CLASES: El modelo espera {len(numerical_classes)} clases numéricas, pero se definieron {len(STRING_RISK_CLASSES)} etiquetas de riesgo en texto.")
            return
    except Exception:
        print("ERROR: No se pudo verificar el número de clases numéricas del modelo.")
        return
    

    
    # Usamos dato_1 como ejemplo predeterminado.
    dato_1 = {
        'Project_Type': 'Construction',
        'Team_Size': 32,
        'Project_Budget_USD': 1526276.55,
        'Estimated_Timeline_Months': 32,
        'Complexity_Score': 9.7,
        'Stakeholder_Count': 16,
        'Methodology_Used': 'Waterfall',
        'Team_Experience_Level': 'Senior',
        'Past_Similar_Projects': 3,
        'External_Dependencies_Count': 3,
        'Change_Request_Frequency': 1.05,
        'Project_Phase': 'Planning',
        'Requirement_Stability': 'Moderate',
        'Team_Turnover_Rate': 0.16,
        'Vendor_Reliability_Score': 0.84,
        'Historical_Risk_Incidents': 2,
        'Communication_Frequency': 1.9,
        'Regulatory_Compliance_Level': 'Medium',
        'Technology_Familiarity': 'Expert',
        'Geographical_Distribution': 4,
        'Stakeholder_Engagement_Level': 'Medium',
        'Schedule_Pressure': 0.0,
        'Budget_Utilization_Rate': 0.82,
        'Executive_Sponsorship': 'Moderate',
        'Funding_Source': 'Government',
        'Market_Volatility': 0.55,
        'Integration_Complexity': 2.66,
        'Resource_Availability': 0.98,
        'Priority_Level': 'Medium',
        'Organizational_Change_Frequency': 0.29,
        'Cross_Functional_Dependencies': 6,
        'Previous_Delivery_Success_Rate': 0.8,
        'Technical_Debt_Level': 0.0,
        'Project_Manager_Experience': 'Mid-level PM',
        'Org_Process_Maturity': 'Managed',
        'Data_Security_Requirements': 'Medium',
        'Key_Stakeholder_Availability': 'Limited',
        'Tech_Environment_Stability': None,  
        'Contract_Type': 'Time & Materials',
        'Resource_Contention_Level': 'High',
        'Industry_Volatility': 'Extreme',
        'Client_Experience_Level': 'First-time',
        'Change_Control_Maturity': 'Basic',
        'Risk_Management_Maturity': 'Basic',
        'Team_Colocation': 'Fully Colocated',
        'Documentation_Quality': 'Good',
        'Project_Start_Month': 10,
        'Current_Phase_Duration_Months': 5,
        'Seasonal_Risk_Factor': 1.0
    }

    dato_2 = {
        'Project_Type': 'Manufacturing',
        'Team_Size': 2,
        'Project_Budget_USD': 390790.15,
        'Estimated_Timeline_Months': 9,
        'Complexity_Score': 2.72,
        'Stakeholder_Count': 9,
        'Methodology_Used': 'Kanban',
        'Team_Experience_Level': 'Mixed',
        'Past_Similar_Projects': 0,
        'External_Dependencies_Count': 2,
        'Change_Request_Frequency': 2.61,
        'Project_Phase': 'Execution',
        'Requirement_Stability': 'Moderate',
        'Team_Turnover_Rate': 0.42,
        'Vendor_Reliability_Score': 0.79,
        'Historical_Risk_Incidents': 2,
        'Communication_Frequency': 2.65,
        'Regulatory_Compliance_Level': 'High',
        'Technology_Familiarity': 'Familiar',
        'Geographical_Distribution': 5,
        'Stakeholder_Engagement_Level': 'Excellent',
        'Schedule_Pressure': 0.0,
        'Budget_Utilization_Rate': 0.76,
        'Executive_Sponsorship': 'Weak',
        'Funding_Source': 'External',
        'Market_Volatility': 0.29,
        'Integration_Complexity': 2.45,
        'Resource_Availability': 0.95,
        'Priority_Level': 'Low',
        'Organizational_Change_Frequency': 0.5,
        'Cross_Functional_Dependencies': 3,
        'Previous_Delivery_Success_Rate': 0.73,
        'Technical_Debt_Level': 0.0,
        'Project_Manager_Experience': 'Mid-level PM',
        'Org_Process_Maturity': 'Optimizing',
        'Data_Security_Requirements': 'Low',
        'Key_Stakeholder_Availability': 'Excellent',
        'Tech_Environment_Stability': None,  # Valor vacío en la fuente
        'Contract_Type': 'Cost-Plus',
        'Resource_Contention_Level': 'Low',
        'Industry_Volatility': 'Stable',
        'Client_Experience_Level': 'Occasional',
        'Change_Control_Maturity': 'Advanced',
        'Risk_Management_Maturity': 'Formal',
        'Team_Colocation': 'Fully Remote',
        'Documentation_Quality': 'Poor',
        'Project_Start_Month': 9,
        'Current_Phase_Duration_Months': 3,
        'Seasonal_Risk_Factor': 1.0
    }

    dato_cliente = {
        # --- FEATURES NUMÉRICAS (Mapeadas del Doc) ---
        'Team_Size': 10,   
        'Project_Budget_USD': 10000.0,
        'Estimated_Timeline_Months': 24,   
        'Complexity_Score': 5,   
        'Stakeholder_Count': 4,   
        'Past_Similar_Projects': 1,   
        'External_Dependencies_Count': 3,   
        'Change_Request_Frequency': 3,   
        'Team_Turnover_Rate': 0.2,   
        'Vendor_Reliability_Score': 0.9,   
        'Historical_Risk_Incidents': 2,
        'Communication_Frequency': 4,   
        'Geographical_Distribution': 2,   
        'Schedule_Pressure': 1.0,   
        'Budget_Utilization_Rate': 0.5, 
        'Market_Volatility': 0.3,   
        'Integration_Complexity': 0,   
        'Resource_Availability': 0.5,
        'Organizational_Change_Frequency': 1, 
        'Cross_Functional_Dependencies': 4,   
        'Previous_Delivery_Success_Rate': 0.5,   
        'Technical_Debt_Level': 0.7,   
        'Project_Start_Month': 9,   
        'Current_Phase_Duration_Months': 14,  
        'Seasonal_Risk_Factor': 1.3,  

        # --- FEATURES CATEGÓRICAS ---
        'Project_Type': 'R&D',
        'Methodology_Used': 'Hybrid',
        'Team_Experience_Level': 'Mixed',
        'Project_Phase': 'Ejecución',
        'Requirement_Stability': 'Moderado',
        'Regulatory_Compliance_Level': 'Crítico',
        'Technology_Familiarity': 'Familiarizados',
        'Stakeholder_Engagement_Level': 'Medio',
        'Executive_Sponsorship': 'Alto/Fuerte',
        'Funding_Source': 'Gobierno',
        'Priority_Level': 'Alto',
        'Project_Manager_Experience': 'Mid-level PM',
        'Org_Process_Maturity': 'Gestionado',
        'Data_Security_Requirements': 'Medio',
        'Key_Stakeholder_Availability': 'Limitado',
        'Tech_Environment_Stability': 'Moderna/ Estable',
        'Contract_Type': 'Precio fijo',
        'Resource_Contention_Level': 'Medio',
        'Industry_Volatility': 'Moderada',
        'Client_Experience_Level': 'Ocasional',
        'Change_Control_Maturity': 'Avanzado',
        'Risk_Management_Maturity': 'Avanzado',
        'Team_Colocation': 'Hibrido',
        'Documentation_Quality': 'Buena'
    }

    # Crear el DataFrame de una sola fila
    new_data = pd.DataFrame([dato_cliente])

    # 1. Predicción
    try:
        prediction_encoded = full_pipeline.predict(new_data)
        probabilities = full_pipeline.predict_proba(new_data)[0]
        
        # 2. Descodificación de la clase predicha (usando la lista de strings)
        predicted_risk = STRING_RISK_CLASSES[prediction_encoded[0]]
        
        print("\n--- RESULTADO DE LA CLASIFICACIÓN ---")
        print(f"Riesgo Previsto: {predicted_risk}")
        print("Probabilidades por Clase:")
        
        # 3. Mostrar las probabilidades mapeadas (usando la lista de strings)
        for i, prob in enumerate(probabilities):
            class_label = STRING_RISK_CLASSES[i]
            print(f"  - {class_label}: {prob:.4f}")

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        print("\nVERIFICA: Los nombres de las columnas en el diccionario de entrada deben coincidir EXACTAMENTE con las features de entrenamiento.")

if __name__ == '__main__':
    main_predict()


'''

Construction,32,1526276.55,32,9.7,16,Waterfall,Senior,3,3,1.05,Planning,Moderate,0.16,0.84,2,1.9,Medium,Expert,4,Medium,0.0,0.82,Moderate,Government,0.55,2.66,0.98,Medium,0.29,6,0.8,0.0,Mid-level PM,Managed,Medium,Limited,,Time & Materials,High,Extreme,First-time,Basic,Basic,Fully Colocated,Good,10,5,1.0,High
Manufacturing,2,390790.15,9,2.72,9,Kanban,Mixed,0,2,2.61,Execution,Moderate,0.42,0.79,2,2.65,High,Familiar,5,Excellent,0.0,0.76,Weak,External,0.29,2.45,0.95,Low,0.5,3,0.73,0.0,Mid-level PM,Optimizing,Low,Excellent,,Cost-Plus,Low,Stable,Occasional,Advanced,Formal,Fully Remote,Poor,9,3,1.0,Low

'''