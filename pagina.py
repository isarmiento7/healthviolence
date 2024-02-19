import streamlit as st
import plotly.express as px

df = px.data.iris()  # Usar un conjunto de datos de ejemplo
fig = px.scatter(df, x="sepal_width", y="sepal_length")
st.plotly_chart(fig)
