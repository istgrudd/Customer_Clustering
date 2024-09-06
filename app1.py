import numpy as np
import pickle
import streamlit as st

# Load the clustering model
with open('model_cluster.pkl', 'rb') as cluster_model_file:
    cluster_model = pickle.load(cluster_model_file)

# Load the scaler model
with open('scaler_cluster.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Modify the prediction function for clustering
def assign_cluster(education, income, kidhome, teenhome, recency, wines, fruits, meat, fish, sweets, gold, 
                   numdealspurchases, numwebpurchases, numcatalogpurchases, numstorepurchases, numwebvisitsmonth,
                   age, status):
    
    # Calculate derived features
    child = kidhome + teenhome
    fam_members = (status + 1) + child
    spend = wines + fruits + meat + fish + sweets + gold
    
    # Combine the inputs into a feature array
    features = np.array([[education, income, kidhome, teenhome, recency, wines, fruits, meat, fish, sweets, gold,
                          numdealspurchases, numwebpurchases, numcatalogpurchases, numstorepurchases, numwebvisitsmonth,
                          age, status, child, fam_members, spend]])

    # Scale the features before passing them to the clustering model
    features_scaled = scaler.transform(features)

    # Assign the cluster
    cluster = cluster_model.predict(features_scaled)

    return cluster[0]  # Return the cluster number

# Streamlit interface for the clustering model
def main():
    st.set_page_config(page_title="Customer Clustering App", page_icon="🛍️", layout="centered")

    st.markdown("<div class='main-title'>Customer Clustering App</div>", unsafe_allow_html=True)

    with st.form("clustering_form"):
        # Collect inputs for all features, using the actual range of values in the dataset
        education = st.slider("Education Level (0=Basic, 1=Graduation, 2=PhD)", min_value=0, max_value=2, step=1)
        income = st.slider("Income (in $)", min_value=1730.0, max_value=113734.0, step=10.0)
        kidhome = st.slider("Number of Kids at Home", min_value=0, max_value=2, step=1)
        teenhome = st.slider("Number of Teens at Home", min_value=0, max_value=2, step=1)
        recency = st.slider("Recency (Days since last purchase)", min_value=0, max_value=99, step=1)
        wines = st.slider("Wines Purchased (in units)", min_value=0, max_value=1493, step=1)
        fruits = st.slider("Fruits Purchased (in units)", min_value=0, max_value=199, step=1)
        meat = st.slider("Meat Purchased (in units)", min_value=0, max_value=1725, step=1)
        fish = st.slider("Fish Purchased (in units)", min_value=0, max_value=259, step=1)
        sweets = st.slider("Sweets Purchased (in units)", min_value=0, max_value=263, step=1)
        gold = st.slider("Gold Purchased (in units)", min_value=0, max_value=362, step=1)
        numdealspurchases = st.slider("Number of Purchases with Discount", min_value=0, max_value=15, step=1)
        numwebpurchases = st.slider("Number of Web Purchases", min_value=0, max_value=27, step=1)
        numcatalogpurchases = st.slider("Number of Catalog Purchases", min_value=0, max_value=28, step=1)
        numstorepurchases = st.slider("Number of Store Purchases", min_value=0, max_value=13, step=1)
        numwebvisitsmonth = st.slider("Number of Web Visits Per Month", min_value=0, max_value=20, step=1)
        age = st.slider("Age", min_value=27, max_value=83, step=1)
        status = st.selectbox("Marital Status (1=Married/Together, 0=Single/Other)", [0, 1])

        submitted = st.form_submit_button("Assign Cluster")

    if submitted:
        # Call the clustering function
        assigned_cluster = assign_cluster(education, income, kidhome, teenhome, recency, wines, fruits, meat, fish, 
                                          sweets, gold, numdealspurchases, numwebpurchases, numcatalogpurchases, 
                                          numstorepurchases, numwebvisitsmonth, age, status)

        # Display the assigned cluster
        st.markdown(f"<div class='result'>The assigned cluster is: {assigned_cluster}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
