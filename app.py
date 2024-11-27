import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo
with open("RandomForestClassifier.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Herramienta de Consulta Rápida Predicción Clientes - FNG")

# Valores mínimos y máximos obtenidos de `describe()`
cobertura_min, cobertura_max = 0, 90

# Iniciar el formulario
with st.form("Formulario_modelo"):
    st.subheader("Ingrese los valores de las variables")

    # Variables categóricas agrupadas
    producto = st.selectbox("Producto", ["MICROCREDITO EMPRESARIAL", "Otros", "UNIDOS POR COLOMBIA MICRO", "GARANTIA EMPRESARIAL MULTIPROPOSITO"])
    oficina = st.selectbox("Oficina", ["Centro", "Costa Pacifica", "Eje Cafetero", "Oriente y Orinoquia", "Caribe"])
    depto_cliente = st.selectbox("Departamento del Cliente", ["Centro", "Costa Pacifica", "Eje Cafetero", "Oriente y Orinoquia", "Caribe"])
    interlocutor = st.selectbox("Interlocutor", ["BANCO AGRARIO", "BANCO CAJA SOCIAL", "BANCO DE BOGOTA", 
                                                 "BANCO MUNDO MUJER S.A.", "BANCO W S.A.", "BANCOLOMBIA", 
                                                 "MIBANCO S.A.", "Otros", "BANCAMIA S.A."])
    calif_credito = st.selectbox("Calificación de Crédito", ["B", "C", "D", "E", "Otros", "A"])

    # En la aplicación, los asesores ingresan valores no normalizados
    Cobertura = st.number_input("% de Cobertura", min_value=float(cobertura_min), max_value=float(cobertura_max), step=0.1)
    
    # Botón para enviar el formulario
    submit_button = st.form_submit_button("Ejecutar Modelo")

# Procesar la predicción
if submit_button:

    # Convertir variables categóricas a formato dummy
    producto_dummy = [1 if producto == "MICROCREDITO EMPRESARIAL" else 0,
                      1 if producto == "Otros" else 0,
                      1 if producto == "UNIDOS POR COLOMBIA MICRO" else 0]
    
    oficina_dummy = [1 if oficina == "Centro" else 0,
                     1 if oficina == "Costa Pacifica" else 0,
                     1 if oficina == "Eje Cafetero" else 0,
                     1 if oficina == "Oriente y Orinoquia" else 0]
    
    depto_cliente_dummy = [1 if depto_cliente == "Centro" else 0,
                           1 if depto_cliente == "Costa Pacifica" else 0,
                           1 if depto_cliente == "Eje Cafetero" else 0,
                           1 if depto_cliente == "Oriente y Orinoquia" else 0]
    
    interlocutor_dummy = [1 if interlocutor == "BANCO AGRARIO" else 0,
                          1 if interlocutor == "BANCO CAJA SOCIAL" else 0,
                          1 if interlocutor == "BANCO DE BOGOTA" else 0,
                          1 if interlocutor == "BANCO MUNDO MUJER S.A." else 0,
                          1 if interlocutor == "BANCO W S.A." else 0,
                          1 if interlocutor == "BANCOLOMBIA" else 0,
                          1 if interlocutor == "MIBANCO S.A." else 0,
                          1 if interlocutor == "Otros" else 0]
    
    calif_credito_dummy = [1 if calif_credito == "B" else 0,
                           1 if calif_credito == "C" else 0,
                           1 if calif_credito == "D" else 0,
                           1 if calif_credito == "E" else 0,
                           1 if calif_credito == "Otros" else 0]

    # Normalizar los valores ingresados
    cobertura_normalizada = (Cobertura - cobertura_min) / (cobertura_max - cobertura_min)

    # Construir el vector de entrada completo
    input_data = np.array([
        *producto_dummy, *oficina_dummy, *depto_cliente_dummy, *interlocutor_dummy,
        *calif_credito_dummy, cobertura_normalizada
    ]).reshape(1, -1)

    # Mostrar los valores de entrada antes de la predicción
    st.write("Valores de entrada al modelo:", input_data)

    # Verificar la longitud del vector de entrada
    if input_data.shape[1] != 25:
        st.error(f"Error: Se esperaban 25 características, pero se obtuvieron {input_data.shape[1]}.")
    else:
        # Realizar la predicción
        prediction = model.predict(input_data)
        pred_value = int(prediction.flatten()[0])
        resultado = "No siniestrada" if pred_value == 0 else "Siniestrada"
        st.write("Predicción del modelo:", resultado)
