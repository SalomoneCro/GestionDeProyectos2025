#Revisemos si hay datos nulos

def valores_nulos(data):
    print("Valores nulos por columna:")
    null = data.isnull().sum()
    return null

#Transformamos nombres de columnas
def transformar_nombres_columnas(data):
    """
    Convierte nombres de mayúsculas a formato estándar:
    -Todo minúscula
    -Espacios reemplazados por guiones bajos
    -Quita caracteres especiales
    """
    data.columns =(data.columns
                   .str.lower())
    print("Columnas renombradas correctamente")
    return data

#Conversión de tipos de datos
def transformar_tipos(data):
    """
    Detecta y transforma tipos de columnas comunes:
    -Fechas a datatime
    -Variables categóricas a category
    """
    for col in data.columns:
        if 'date' in col:
            data[col] = pd.to_datetime(data[col], errors="coerce")
        elif data[col].dtype == 'object' and data[col].nunique() < 50:
            data[col] = data[col].astype('category')
    print("Tipos de datos transformados correctamente")
    return data

#Ahora realizamos la limpieza completa
def limpiar_y_transformar(data):
    print("→ Valores nulos por columna:")
    print(valores_nulos(data), "\n")

    print("→ Transformando nombres de columnas:")
    data = transformar_nombres_columnas(data)  # guardamos el resultado modificado

    print("→ Transformando tipos de datos:")
    data = transformar_tipos(data)

    print("\n Limpieza y transformación completadas.")
    return data

limpiar_y_transformar(data)
