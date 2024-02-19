import streamlit as st
import pandas as pd
import json
#import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def load_geojson(path):
    with open(path) as f:
        return json.load(f)

def load_delitos_data(path):
    return pd.read_stata(path)

def prepare_data(geojson, delitos_df):
    for feature in geojson['features']:
        feature['id'] = feature['properties']['cod_comuna']

    comunas_data = [{'cod_comuna': feature['properties']['cod_comuna'],
                     'nombre_comuna': feature['properties']['Comuna']} for feature in geojson['features']]
    comunas_df = pd.DataFrame(comunas_data)

    delitos_df = delitos_df.merge(comunas_df, left_on='comuna', right_on='cod_comuna')
    return delitos_df

def generate_map(delitos_df, geojson, var):
    fig = px.choropleth_mapbox(delitos_df, geojson=geojson, locations='nombre_comuna', color=var,
                               featureidkey="properties.Comuna",
                               center={"lat": -33.6751, "lon": -71.5430},
                               color_continuous_scale='Magma_r',
                               mapbox_style="carto-positron", zoom=5.5)

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

# Sidebar
st.sidebar.header('Índice')
variable_seleccionada = st.sidebar.selectbox('Selecciona una página:', ["Estadísticas por variables","Cruces de datos","Mapas principales","Mapas delitos","Regresiones","Cuestionarios"])

geojson_path = 'comunas.geojson'  # Asegúrate de actualizar esta ruta
reg_path = 'Base_regresiones.dta'
reg_path2 = 'Base_regresiones2.dta'  # Asegúrate de actualizar esta ruta
df_reg = load_delitos_data(reg_path)



if variable_seleccionada=="Estadísticas por variables":
    st.title('Estadísticas por variables')
    

    st.sidebar.header('Selección de Variable')
    vs1 = {
        "Índice percepción delincuencia": "h4i",
        "Satisfacción seguridad": "satisfaccion",
        "Seguridad plazas y parques":"h4_1",
        "Seguridad caminando de día":"h4_2",
        "Seguridad caminando de noche":"h4_3",
        "Seguridad en casa":"h4_4",
        "Depresión": "depre",
        "Ansiedad":"ansiedad",
        "Distress":"distress",
        "Region":"region",
        "Sexo":"sexo",
        "Educacion en tramo":"educacion_tramo",
        "Edad en tramos":"edad_tramo",
        "Ingreso en tramo":"ingreso_tramo",
        "Escolaridad":"esc",
        "Ingreso":"ytotcorh",
        "Zona":"zona",
        "Tiene amigos":"amigos",
        "Uso redes sociales":"social",
        "Edad":"edad",
        "Problemas mentales diagnosticados":"mentah_dia",
        "Tiene conviviente": "conviviente",
        "Victima de delitos": "victima"

        }    

    opcion_variable = st.sidebar.selectbox('Elige la variable:', list(vs1.keys()))

    ns1 = vs1[opcion_variable]


    st.write(f"Estadísticas descriptivas de {opcion_variable}:")
    st.table(df_reg[ns1].describe().T)
    fig, ax = plt.subplots()
    if df_reg[ns1].dtype in ['int64', 'float64']:  # Para variables numéricas
        sns.histplot(df_reg[ns1], kde=True, ax=ax)
        st.write(f"Distribución de {opcion_variable}:")
    else:  # Para variables categóricas
        sns.countplot(x=ns1, data=df_reg, ax=ax)
        st.write(f"Distribución de frecuencias de {opcion_variable}:")
    st.pyplot(fig)
    

if variable_seleccionada=='Cruces de datos':
    st.title('Cruces de variables')
    vs1 = {
        "Índice percepción": "h4i",
        "Depresión": "depre",
        "Ansiedad":"ansiedad",
        "Distress":"distress",
        "Escolaridad":"esc",
        "Ingreso":"ytotcorh",
        "Tiene amigos":"amigos",
        "Uso redes sociales":"social",
        "Edad":"edad",
        "Problemas mentales diagnosticados":"mentah_dia",
        "Tiene conviviente": "conviviente",
        "Victima de delitos": "victima"

        }    
    vs2 = {

        "Depresión": "depre",
        "Ansiedad":"ansiedad",
        "Distress":"distress",
        "Region":"region",
        "Sexo":"sexo",
        "Educacion en tramo":"educacion_tramo",
        "Edad en tramos":"edad_tramo",
        "Ingreso en tramo":"ingreso_tramo",
        "Zona":"zona",
        "Tiene amigos":"amigos",
        "Uso redes sociales":"social",
        "Problemas mentales diagnosticados":"mentah_dia",
        "Tiene conviviente": "conviviente",
        "Victima de delitos": "victima",
        "Satisfacción seguridad": "satisfaccion",
        "Seguridad plazas y parques":"h4_1",
        "Seguridad caminando de día":"h4_2",
        "Seguridad caminando de noche":"h4_3",
        "Seguridad en casa":"h4_4"

        }  
    opcion_variable1 = st.sidebar.selectbox('Elige la primera variable:', list(vs1.keys()), key='var1')
    opcion_variable2 = st.sidebar.selectbox('Elige la segunda variable (Categorias):', list(vs2.keys()), key='var2')

    # Obtener los nombres de las columnas del DataFrame basados en la selección del usuario
    ns2 = vs1[opcion_variable1]
    ns1 = vs2[opcion_variable2]
    st.write(f"Distribución de {opcion_variable1} por {opcion_variable2}")    

    # Calcular promedios
    promedios = df_reg.groupby(ns1)[ns2].mean().reset_index()

    # Gráfico de barras
    st.write("Gráfico de barras")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(x=ns1, y=ns2, data=promedios, ax=ax)
    ax.set_title(f'Promedio de {opcion_variable1} por {opcion_variable2}')
    ax.set_xlabel(opcion_variable1)
    ax.set_ylabel(f'Promedio de {opcion_variable2}')
    st.pyplot(fig)

    # Mostrar la tabla de promedios
    # Formatear la tabla para mostrar valores numéricos con dos decimales
    tabla_formateada = promedios.style.format({ns2: "{:.2f}"})
    
    # Mostrar la tabla formateada en Streamlit
    st.write("Promedios")
    st.table(tabla_formateada)



    # Inicializa la figura de Matplotlib
    fig, ax = plt.subplots()

    # Gráfico de cajas
    sns.boxplot(x=ns1, y=ns2, data=df_reg, ax=ax)
    st.write("Gráfico de cajas")
    st.pyplot(fig)


if variable_seleccionada=="Mapas principales":
    # Streamlit app
    st.title('Mapas comparativos')
    
    
    delitos_path2 = 'C:/Users/isarm/Desktop/Salud mental y violencia/Bases/Base_mapas.dta'  # Asegúrate de actualizar esta ruta
    
    geojson = load_geojson(geojson_path)
    delitos_df = load_delitos_data(delitos_path2)
    #delitos_df=delitos_df.rename(columns={"comuna":"nombre","cod_comuna":"comuna"})
    delitos_df = prepare_data(geojson, delitos_df)
    
    
    vm1 = {
        "Índice percepción violencia": "h4i",
        "Satisfacción seguridad": "satisfaccion",
        "Depresión": "depre",
        "Ansiedad":"ansiedad",
        "Distress":"distress",
        "Escolaridad":"esc",
        "Uso redes sociales":"social",
        "Edad":"edad",
        "Problemas mentales diagnosticados":"mental_dia"
        "Pobreza:pobres",
        "Total DMCS": "total_dmcs",
        "Robo con violencia": "rb_violencia",
        "Robo por sorpresa": "rb_sorpresa",
        "Robo con fuerza":"rb_fuerza",
        "Robo vehiculo":"rb_vehiculo",
        "Robo accesorios vehiculo":"rb_acc_vehi",
        "Robo lugar habtiado":"rb_lugar_hab",
        "Robo lugar no habtiado":"rb_lugar_nohab",
        "Hurtos":"hurtos",
        "Lesiones":"lesiones",
        "Violaciones":"violaciones",
        "Homicidios":"homicidio"
        "Pobreza:pobres"
        }
        
    vm2 = {
        "Satisfacción seguridad": "satisfaccion",
        "Depresión": "depre",
        "Ansiedad":"ansiedad",
        "Distress":"distress",
        "Escolaridad":"esc",
        "Uso redes sociales":"social",
        "Edad":"edad",
        "Problemas mentales diagnosticados":"mental_dia"
        "Pobreza:pobres",
        "Total DMCS": "total_dmcs",
        "Robo con violencia": "rb_violencia",
        "Robo por sorpresa": "rb_sorpresa",
        "Robo con fuerza":"rb_fuerza",
        "Robo vehiculo":"rb_vehiculo",
        "Robo accesorios vehiculo":"rb_acc_vehi",
        "Robo lugar habtiado":"rb_lugar_hab",
        "Robo lugar no habtiado":"rb_lugar_nohab",
        "Hurtos":"hurtos",
        "Lesiones":"lesiones",
        "Violaciones":"violaciones",
        "Homicidios":"homicidio"
        "Pobreza:pobres"
        }
    
    
    mod1 = st.selectbox(
        'Seleccione la primera variable:',
        list(vm1.keys())
        )


    mod2 = st.selectbox(
        'Seleccione la segunda variable:',
        list(vm2), key="m2"
        )
    
    varmod1 = vm1[mod1]
    varmod2 = vm2[mod2]

    
    fig1 = generate_map(delitos_df, geojson,varmod1)
    fig2 = generate_map(delitos_df, geojson,varmod2)
    
    st.header(mod1)
    st.plotly_chart(fig1, use_container_width=True)
    st.header(mod2)
    st.plotly_chart(fig2, use_container_width=True)

    datos_o1 = delitos_df[["nombre_comuna",varmod1]].sort_values(by=varmod1, ascending=False)     
    tabla1 = datos_o1.head(10)
    datos_o2 = delitos_df[["nombre_comuna",varmod2]].sort_values(by=varmod2, ascending=False)  
    tabla2 = datos_o2.head(10)
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Top 10 comuna "+mod1)
        st.table(tabla1["nombre_comuna"])
    
    with col2:
        st.header("Top 10 comuna "+mod2)
        st.table(tabla2["nombre_comuna"])
        
    # Calcular la correlación
    correlacion = delitos_df[varmod1].corr(delitos_df[varmod2])
    
    # Mostrar la correlación en Streamlit
    st.write(f'La correlación entre {mod1} y {mod2} es de {correlacion:.2f}')
    

if variable_seleccionada=="Mapas delitos":
    # Streamlit app
    st.title('Mapa de Tasas de Delitos por Comuna por año')
    
    
    delitos_path = 'C:/Users/isarm/Desktop/Salud mental y violencia/Bases/tasas_delitos.dta'  # Asegúrate de actualizar esta ruta
    
    geojson = load_geojson(geojson_path)
    delitos_df = load_delitos_data(delitos_path)
    delitos_df=delitos_df.rename(columns={"comuna":"nombre","cod_comuna":"comuna"})
    delitos_df = prepare_data(geojson, delitos_df)
    
    
    var_mod = {
        "Total DMCS": "total_dmcs",
        "Robo con violencia": "rb_violencia",
        "Robo por sorpresa": "rb_sorpresa",
        "Robo con fuerza":"rb_fuerza",
        "Robo vehiculo":"rb_vehiculo",
        "Robo accesorios vehiculo":"rb_acc_vehi",
        "Robo lugar habtiado":"rb_lugar_hab",
        "Hurtos":"hurtos",
        "Lesiones":"lesiones",
        "Violaciones":"violaciones",
        "Homicidios":"homicidio"
        
        }
        
    var_mod2=var_mod    
    
    modulo1 = st.selectbox(
        'Seleccione un tipo de delito:',
        list(var_mod.keys())
        )
    
    var_sel1 = var_mod[modulo1]+"_2019"
    var_sel2 = var_mod2[modulo1]+"_2020"
    var_sel3 = var_mod2[modulo1]+"_2021"
    var_sel4 = var_mod2[modulo1]+"_2022"
    
    fig1 = generate_map(delitos_df, geojson,var_sel1)
    fig2 = generate_map(delitos_df, geojson,var_sel2)
    fig3 = generate_map(delitos_df, geojson,var_sel3)
    fig4 = generate_map(delitos_df, geojson,var_sel4)
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("2019")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.header("2020")
        st.plotly_chart(fig2, use_container_width=True)
     
    col3, col4 = st.columns(2)
    with col3:
        st.header("2021")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.header("2022")
        st.plotly_chart(fig4, use_container_width=True)
    

if variable_seleccionada=="Cuestionarios":
    st.title('Cuestionarios')
    st.markdown('[Encuesta de bienestar social 2021](https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/bienestar-social/Cuestionario_Encuesta_Bienestar_Social_2021.pdf)')
    st.markdown('[Encuesta CASEN 2020](https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2020/Cuestionario_Casen_En_Pandemia_2020.pdf)')
        
    


if variable_seleccionada=="Regresiones":
    df_reg = load_delitos_data(reg_path2)
    import statsmodels.api as sm
    st.title('Regresiones')    

    regmod = st.sidebar.selectbox('Seleccione un modelo:', ["MCO","Probit","Variable Instrumental", "Random Forest"])


    dep = {
        "Distress":"distress",
        "Depresión": "depre",
        "Ansiedad":"ansiedad"
        }  
    
    indep = {
        "Índice percepción": "h4i",
        "Satisfacción seguridad": "satisfaccion",
        "Seguridad plazas y parques":"h4_1",
        "Seguridad caminando de día":"h4_2",
        "Seguridad caminando de noche":"h4_3",
        "Seguridad en casa":"h4_4"
        }
    
    controles = {
        "Sexo":"sexo",
        "Educacion en tramos":"educacion_tramo",
        "Edad en tramos":"edad_tramo",
        "Edad":"edad",
        "Ingreso en tramos":"ingreso_tramo",
        "Ingreso":"ytotcorh",
        "Zona":"zona",
        "Tiene amigos":"amigos",
        "Uso redes sociales":"social",
        "Problemas mentales":"mentah_dia",
        "Problemas fisicos":"phy_status",
        "Tiene conviviente": "conviviente",
        "Victima de delitos": "victima"

        }    

    instrumentos = {
        "Promedio Índice":"ch4i",
        "Total Delitos MCS":"total_dmcs",
        "Robo con violencia":"rb_violencia",
        "Robo con sorpresa":"rb_sorpresa",
        "Robo con fuerza":"rb_fuerza",
        "Violaciones":"violaciones",
        "Tasa de pobreza":"pobres"

        }      
    
    
    
    #seleccion dependiente

    sel_dep = st.sidebar.selectbox('Elige la variable dependiente:', list(dep.keys()), key='var1')
    st.write('Has seleccionado:', sel_dep)

    valor_predeterminado=["Índice percepción"]

    
    
    
    
    # Crear el objeto de selección múltiple en la aplicación Streamlit
    if regmod!="Variable Instrumental":  
        selecciones = st.sidebar.multiselect('Selecciona la(s) variable(s) independiente(s) principal(es):', list(indep.keys()), key='reg1', default=valor_predeterminado)
        var_indep = [indep[seleccion] for seleccion in selecciones]   
        
    if regmod=="Variable Instrumental":  
        var_indep = st.sidebar.selectbox('Selecciona la(s) variable(s) independiente(s) principal(es):', list(indep.keys()), key='reg8')

        
    if regmod=="Variable Instrumental" or regmod=="Probit IV":    
        selecciones_ins = st.sidebar.multiselect('Selecciona los instrumentos:', list(instrumentos.keys()), key='reg3', default=["Promedio Índice"])
        var_ins = [instrumentos[seleccion] for seleccion in selecciones_ins]
    
  
    pre_controles=["Sexo","Ingreso en tramos","Edad en tramos","Educacion en tramos","Zona","Tiene amigos","Tiene conviviente","Victima de delitos","Uso redes sociales","Problemas fisicos","Problemas mentales"]
    seleccionesc = st.sidebar.multiselect('Selecciona la(s) variable(s) de control:', list(controles.keys()), key='reg2', default=pre_controles)
    

    var_controles = [controles[seleccion] for seleccion in seleccionesc]
    
    tramos = ["ingreso_tramo","educacion_tramo","zona", "edad_tramo"]
    
    

    
    
    # Definir la variable dependiente y las independientes
    y = df_reg[dep[sel_dep]]
    if regmod!="Variable Instrumental":  
        X = pd.concat([df_reg[var_indep], df_reg[var_controles]], axis=1)
    if regmod=="Variable Instrumental":  
        X = pd.concat([df_reg[indep[var_indep]], df_reg[var_controles]], axis=1)
 
    
    for t in tramos:
        if t in var_controles:
            dummies = pd.get_dummies(df_reg[t], prefix=t, drop_first=True).astype(int)
            X = X.join(dummies).drop(columns=t)

    X = sm.add_constant(X) 
    X = sm.add_constant(X)  # Añadir una constante al modelo
    
    if regmod=="MCO":
        # Ejecutar la regresión MCO
        modelo = sm.OLS(y, X).fit()
        
        # Crear un DataFrame con los resultados
        resultados = pd.DataFrame({
            'Coeficientes': modelo.params,
            'Errores Estándar': modelo.bse,
            'Valores t': modelo.tvalues,
            'P-valores': modelo.pvalues
        })
        
        # Añadir asteriscos para indicar significancia
        significancia = []
        for p in resultados['P-valores']:
            if p < 0.01:
                significancia.append('***')
            elif p < 0.05:
                significancia.append('**')
            elif p < 0.1:
                significancia.append('*')
            else:
                significancia.append('')
        
        resultados['Significancia'] = significancia
        
        # Opcional: Formatear la columna de P-valores para mejorar la legibilidad
        resultados['P-valores'] = resultados['P-valores'].map('{:.4f}'.format)
        
        st.write('Resultados de la Regresión MCO:')
        st.table(resultados)
        
        
    if regmod=="Probit":
        # Ejecutar la regresión MCO
        modelo = sm.Probit(y, X).fit()
        
        # Crear un DataFrame con los resultados
        resultados = pd.DataFrame({
            'Coeficientes': modelo.params,
            'Errores Estándar': modelo.bse,
            'Valores t': modelo.tvalues,
            'P-valores': modelo.pvalues
        })
        
        # Añadir asteriscos para indicar significancia
        significancia = []
        for p in resultados['P-valores']:
            if p < 0.01:
                significancia.append('***')
            elif p < 0.05:
                significancia.append('**')
            elif p < 0.1:
                significancia.append('*')
            else:
                significancia.append('')
        
        resultados['Significancia'] = significancia
        
        # Opcional: Formatear la columna de P-valores para mejorar la legibilidad
        resultados['P-valores'] = resultados['P-valores'].map('{:.4f}'.format)
        
        st.write('Resultados de la Regresión MCO:')
        st.table(resultados)
        efectos_marginales = modelo.get_margeff()
        st.text(efectos_marginales.summary().as_text())
    
    if regmod=="Variable Instrumental":
        from linearmodels.iv import IV2SLS
#        from linearmodels.iv import IVProbit
        
#        st.table(indep[var_indep])
        # Definir las variables endógenas, exógenas e instrumentales
        endog = df_reg[indep[var_indep]]  # La variable que estás instrumentando
        exog = X.drop(columns=[indep[var_indep]]) # Variables exógenas (controles + dummies)
        instr = df_reg[var_ins]  # Variables instrumentales para 'h4i'
        
        
        # Especificar y ajustar el modelo
        modelo_iv = IV2SLS(dependent=y,  # Variable dependiente
                           exog=exog,
                           endog=endog,
                           instruments=instr).fit(cov_type='robust')
        
        # Crear un DataFrame con los resultados de la segunda etapa
        resultados_iv = pd.DataFrame({
            'Coeficientes': modelo_iv.params,
            'Errores Estándar': modelo_iv.std_errors,
            'Valores t': modelo_iv.tstats,
            'P-valores': modelo_iv.pvalues
        })
        
        # Añadir asteriscos para indicar significancia
        significancia_iv = []
        for p in resultados_iv['P-valores']:
            if p < 0.01:
                significancia_iv.append('***')
            elif p < 0.05:
                significancia_iv.append('**')
            elif p < 0.1:
                significancia_iv.append('*')
            else:
                significancia_iv.append('')
        
        resultados_iv['Significancia'] = significancia_iv
        
        # Opcional: Formatear la columna de P-valores para mejorar la legibilidad
        resultados_iv['P-valores'] = resultados_iv['P-valores'].map('{:.4f}'.format)
        
        st.text(modelo_iv.summary.as_text())

        st.text(modelo_iv.sargan)

        # Mostrar los resultados
        st.table(resultados_iv)
        
        import statsmodels.api as sm

        # Preparar los datos para la primera etapa
        X_primera_etapa = sm.add_constant(pd.concat([exog, instr], axis=1))  # Unir las variables exógenas y las instrumentales
        
        # Ajustar el modelo de la primera etapa
        modelo_primera_etapa = sm.OLS(endog, X_primera_etapa).fit()
        
        # Mostrar los resultados de la primera etapa
        st.text(modelo_primera_etapa.summary().as_text())


    if regmod=="Random Forest":
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        
        # Supongamos que tus datos están en un DataFrame de pandas llamado df
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Inicializar el modelo de Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Entrenar el modelo con los datos de entrenamiento
        rf.fit(X_train, y_train)
        
        # Predecir los resultados para el conjunto de prueba
        y_pred = rf.predict(X_test)
        
        # Evaluar el modelo
        accuracy = accuracy_score(y_test, y_pred)
        st.text(f'Accuracy: {accuracy*100:.2f}% (El modelo Random Forest tiene este % de predicción)')
        
        # Mostrar la importancia de las características
        importances = rf.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.table(importance_df)
