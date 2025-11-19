# convertir_parquet_a_csv.py

import pandas as pd
import os

# --- Configuracion de Rutas ---
# *** ¡CAMBIA ESTO POR EL NOMBRE REAL DE TU ARCHIVO! ***
INPUT_FILE = 'train_ready.parquet' 
OUTPUT_FILE = 'sel_all.csv' 
COLUMN_TO_DROP = 'Project_ID'

def convertir_archivo():
    """Carga un archivo Parquet, elimina la columna Project_ID y lo guarda como CSV."""
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: El archivo de entrada '{INPUT_FILE}' no se encontró.")
        print("Asegúrate de cambiar 'tu_dataset.parquet' por el nombre de tu archivo.")
        return

    print(f"Cargando datos desde: {INPUT_FILE}...")
    try:
        # 1. Cargar el archivo Parquet
        df = pd.read_parquet(INPUT_FILE)
        print(f"✅ Archivo Parquet cargado. Filas: {len(df)}, Columnas: {len(df.columns)}")
        
        # 2. Eliminar la columna Project_ID
        if COLUMN_TO_DROP in df.columns:
            df = df.drop(columns=[COLUMN_TO_DROP])
            print(f"✅ Columna '{COLUMN_TO_DROP}' eliminada exitosamente.")
        else:
            print(f"⚠️ Advertencia: La columna '{COLUMN_TO_DROP}' no se encontró en el archivo.")
        
        # 3. Guardar el DataFrame modificado como CSV
        # El archivo resultante 'sel_all.csv' estará listo para el entrenamiento.
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✨ Conversión completada exitosamente.")
        print(f"Archivo CSV guardado como: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Ocurrió un error durante la conversión: {e}")

if __name__ == '__main__':
    convertir_archivo()