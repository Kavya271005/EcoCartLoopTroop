import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("train.csv")  # or your cleaned DF

# Load model & other artifacts
model = joblib.load('personality_predictor.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

# Set up UI
st.set_page_config(page_title="EcoCart Segmentation", layout="wide")
st.title("🌍 EcoCart Solutions - Customer Segmentation App")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 EDA", "📋 Persona Strategies", "🔍 Clustering Validation"])

# ----------------------------- Tab 1: Prediction -----------------------------
with tab1:
    st.header("Predict Customer Persona")

    # User input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", ["Male", "Female"])
            bachelor = st.selectbox("Marital Status", ["Yes", "No"])
            age = st.slider("Age", 18, 90, 30)
            graduated = st.selectbox("Graduated", ["Yes", "No"])
        with col2:
            career = st.selectbox("Career", ['Doctor', 'Lawyer', 'Scientist', 'Engineer', 'Artist', 'Singer', 'ContentCreation', 'FashionDesigner', 'HouseWife', 'HR'])
            work_exp = st.slider("Work Experience", 0, 40, 5)
            fam_exp = st.selectbox("Family Expenses", ["Low", "Average", "High"])
            fam_size = st.slider("Family Size", 1, 10, 4)

        variable = st.selectbox("Other Variable", ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Dog_6', 'Dog_7', 'Cat_1'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input
        input_df = pd.DataFrame({
            'Sex': [sex], 'Bachelor': [bachelor], 'Age': [age], 'Graduated': [graduated],
            'Career': [career], 'Work Experience': [work_exp], 'Family Expenses': [fam_exp],
            'Family  Size': [fam_size], 'Variable': [variable]
        })

        input_df['Family Expenses'] = input_df['Family Expenses'].map({'Low': 0, 'Average': 1, 'High': 2})
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)
        predicted_persona = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎯 Predicted Persona: **{predicted_persona}**")

# ----------------------------- Tab 2: EDA -----------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")

    st.markdown("### 1. Age Distribution by Persona")
    fig1 = px.histogram(df, x='Age', color='Segmentation', barmode='overlay', nbins=30)
    st.plotly_chart(fig1)
    st.info("🧠 **Insight:** Bhavesh users tend to be older — suggesting settled, family-oriented individuals.")

    st.markdown("### 2. Family Expenses by Persona")
    fig2 = px.box(df, x='Segmentation', y='Family Expenses', color='Segmentation')
    st.plotly_chart(fig2)
    st.info("💡 **Insight:** Chaitanya and Darsh have high family expenses — target them with premium offers.")


    st.markdown("### 3. Marital Status vs. Persona")
    fig3 = px.histogram(df, x='Segmentation', color='Bachelor', barmode='group')
    st.plotly_chart(fig3)
    st.info("🔍 **Insight:** Bhavesh has more married users — ideal for family bundles and loyalty programs.")

    st.markdown("### 4. Work Experience vs. Persona")
    fig4 = px.violin(df, x='Segmentation', y='Work Experience', color='Segmentation', box=True)
    st.plotly_chart(fig4)
    st.info("📈 **Insight:** Akshat and Bhavesh have users with longer experience — consider job security–based finance deals.")

    st.markdown("### 5. Gender Ratio by Persona")
    fig5 = px.histogram(df, x='Segmentation', color='Sex', barmode='group')
    st.plotly_chart(fig5)
    st.info("👩‍🦰 **Insight:** Gender distributions may help tailor marketing channels (e.g., male-dominated personas via YouTube, female via Instagram).")

    df = pd.read_csv("train.csv")
    df['Family Expenses'] = df['Family Expenses'].map({'Low': 0, 'Average': 1, 'High': 2})

    if 'Segmentation' in df.columns:
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Segmentation', ax=ax1)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='Segmentation', y='Age', ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Segmentation column not found. Please upload cleaned data.")

# ----------------------------- Tab 3: Persona Analysis -----------------------------

with tab3:
    st.header("🎯 Persona-Wise Business Strategies")
    
    strategies = {
        "Akshat": {
            "Product Focus": "Budget electronics and gadgets",
            "Channels": "Instagram, WhatsApp",
            "Loyalty Program": "Cashback coupons for young buyers"
        },
        "Bhavesh": {
            "Product Focus": "Home/family essentials, baby products",
            "Channels": "Facebook, Email",
            "Loyalty Program": "Referral bonuses for families"
        },
        "Chaitanya": {
            "Product Focus": "Trendy fashion, career tools",
            "Channels": "LinkedIn, Instagram Ads",
            "Loyalty Program": "Refer & Earn + Internship vouchers"
        },
        "Darsh": {
            "Product Focus": "Wellness products, groceries",
            "Channels": "Email, App Notifications",
            "Loyalty Program": "Senior loyalty club benefits"
        }
    }

    for persona, details in strategies.items():
        st.subheader(f"🧑 {persona}")
        st.markdown(f"""
        - **Product Focus:** {details['Product Focus']}  
        - **Preferred Channels:** {details['Channels']}  
        - **Loyalty Program:** {details['Loyalty Program']}
        """)
        st.markdown("---")




# ----------------------------- Tab 4: Clustering Validation -----------------------------
with tab4:
    st.subheader("🔍 Cluster Validation (KMeans vs Personas)")

    # 1. Prepare numeric features for clustering
    st.write("Preparing data for KMeans clustering...")
    df_cluster = df.copy()
    df_cluster = df_cluster.select_dtypes(include=['int64', 'float64'])
    df_cluster = df_cluster.dropna()
    
    X_cluster = df_cluster.values
    df_copy = df_cluster.copy()

    # 2. Apply KMeans
    st.write("Applying KMeans Clustering (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_copy['Cluster'] = kmeans.fit_predict(X_cluster)

    # 3. Compare Cluster vs. Actual Segment
    st.write("🔁 Comparing Clusters with Personas")
    if 'Segmentation' in df.columns:
        comparison = pd.crosstab(df_copy['Cluster'], df['Segmentation'].iloc[df_copy.index])
        st.dataframe(comparison)

    # 4. Silhouette Score
    st.write("📈 Silhouette Score (Higher is better)")
    score = silhouette_score(X_cluster, df_copy['Cluster'])
    st.success(f"Silhouette Score: {score:.3f}")

    # 5. Elbow Method Plot
    st.write("📉 Elbow Method (to find best k)")
    sse = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_cluster)
        sse.append(km.inertia_)
    
    fig1, ax1 = plt.subplots()
    ax1.plot(K_range, sse, marker='o')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xlabel('k')
    ax1.set_ylabel('SSE (Inertia)')
    st.pyplot(fig1)

    # 6. PCA for Visualization
    st.write("🎯 PCA 2D View of Clusters")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df_copy['Cluster'], cmap='Set1', s=60)
    ax2.set_title("KMeans Clusters (PCA Projection)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    st.pyplot(fig2)

# ----------------------------- Sidebar: Batch Upload -----------------------------
st.sidebar.title("📂 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    batch_df['Family Expenses'] = batch_df['Family Expenses'].map({'Low': 0, 'Average': 1, 'High': 2})
    batch_encoded = pd.get_dummies(batch_df)
    batch_encoded = batch_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(batch_encoded)
    batch_df['Predicted Persona'] = label_encoder.inverse_transform(predictions)
    st.sidebar.success("✅ Batch Prediction Complete")
    st.sidebar.write(batch_df)

    # Download
    st.sidebar.download_button("Download Predictions", batch_df.to_csv(index=False), file_name="batch_predictions.csv")

