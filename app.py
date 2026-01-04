import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Iris ML App",
    page_icon="🌸",
    layout="wide"
)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

model = joblib.load("random_forest_iris_model.pkl")

df = pd.DataFrame(X, columns=feature_names)
df["species"] = y
df["species_name"] = df["species"].apply(lambda x: target_names[x])

st.sidebar.title("🌿 Input Features")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), 5.1
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), 3.5
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), 1.4
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), 0.2
)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

st.title("🌸 Iris Flower Classification App")
st.caption("Random Forest • Streamlit • Plotly • ML")

col1, col2, col3 = st.columns(3)

col1.metric("🌼 Prediction", target_names[prediction])
col2.metric("🔥 Confidence", f"{np.max(prediction_proba)*100:.2f}%")
col3.metric("📦 Model", "Random Forest")

tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🧠 Model Confidence", "📈 Dataset Overview"])

with tab1:
    st.subheader("2D Feature Relationship")

    fig_2d = px.scatter(
        df,
        x="petal length (cm)",
        y="petal width (cm)",
        color="species_name",
        opacity=0.7
    )
    fig_2d.add_scatter(
        x=[petal_length],
        y=[petal_width],
        mode="markers",
        marker=dict(size=15, color="black", symbol="star"),
        name="Your Input"
    )

    st.plotly_chart(fig_2d, use_container_width=True)

    st.subheader("3D Iris Visualization")

    fig_3d = px.scatter_3d(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        z="petal length (cm)",
        color="species_name",
        opacity=0.6
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=[sepal_length],
            y=[sepal_width],
            z=[petal_length],
            mode="markers",
            marker=dict(size=8, color="black", symbol="diamond"),
            name="Your Input"
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Prediction Probability")

    proba_df = pd.DataFrame({
        "Species": target_names,
        "Probability": prediction_proba
    })

    fig_bar = px.bar(
        proba_df,
        x="Species",
        y="Probability",
        color="Species",
        text_auto=".2f"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("Iris Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Feature Distribution")
    feature = st.selectbox("Choose Feature", feature_names)

    fig_hist = px.histogram(
        df,
        x=feature,
        color="species_name",
        marginal="box"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("Built with 💻 Streamlit & ML | By Uzair")
