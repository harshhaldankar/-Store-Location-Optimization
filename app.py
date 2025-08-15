import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

# Page config
st.set_page_config(page_title="Store Location Optimization", page_icon="🏪", layout="wide")

# Cache expensive computations
@st.cache_data
def load_and_process_data(uploaded_file):
    """Cache data loading and initial processing"""
    data = pd.read_csv(uploaded_file, index_col=0)
    return data

@st.cache_data
def perform_clustering(x, n_clusters, random_state=10):
    """Cache clustering results"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    y_kmeans = kmeans.fit_predict(x)
    return kmeans, y_kmeans

@st.cache_data
def calculate_elbow_method(x, max_clusters=10):
    """Cache elbow method calculation"""
    wsse = []
    k_range = range(1, max_clusters + 1)
    
    progress_bar = st.progress(0)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(x)
        wsse.append(kmeans.inertia_)
        progress_bar.progress((i + 1) / len(k_range))
    
    progress_bar.empty()
    return k_range, wsse

# Title
st.title("Store Location Optimization")

# Sidebar parameters
st.sidebar.header("Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
density_threshold = st.sidebar.slider("Density Threshold (%)", 5, 50, 10)
percentile_coverage = st.sidebar.slider("Percentile Coverage (%)", 80, 99, 95)

# Performance options
st.sidebar.header("Performance Options")
show_elbow = st.sidebar.checkbox("Show Elbow Method (slower)", value=False)
max_elbow_clusters = st.sidebar.slider("Max Clusters for Elbow", 5, 15, 8) if show_elbow else 8

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data with caching
        with st.spinner("Loading data..."):
            data = load_and_process_data(uploaded_file)
        
        st.success(f"Data loaded: {len(data)} orders")
        
        # Quick data overview
        st.info(f"**Data Summary:** {len(data)} orders | Lat: {data['Lat'].min():.4f} to {data['Lat'].max():.4f} | Long: {data['Long'].min():.4f} to {data['Long'].max():.4f}")
        
        # Prepare data for clustering
        x = data[['Lat', 'Long']].values
        
        # Original clustering (100% coverage)
        st.header("Clustering Analysis")
        
        # Elbow method (optional for performance)
        if show_elbow:
            st.subheader("Elbow Method - Optimal Number of Clusters")
            with st.spinner("Calculating optimal clusters (this may take a moment)..."):
                k_range, wsse = calculate_elbow_method(x, max_elbow_clusters)
            
            # Elbow plot
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(k_range), y=wsse, mode='lines+markers'))
            fig_elbow.update_layout(title="Elbow Method - Optimal Number of Clusters", height=400)
            st.plotly_chart(fig_elbow, use_container_width=True)
        else:
            st.info("**Tip:** Enable 'Show Elbow Method' in sidebar for optimal cluster analysis")
        
        # Perform clustering with caching
        with st.spinner("Performing clustering..."):
            kmeans_original, y_kmeans_original = perform_clustering(x, n_clusters)
            
            # Add cluster results
            data['cluster_original'] = y_kmeans_original
            data['distance_original'] = data.apply(
                lambda row: np.sqrt(
                    (row['Lat'] - kmeans_original.cluster_centers_[int(row['cluster_original'])][0])**2 +
                    (row['Long'] - kmeans_original.cluster_centers_[int(row['cluster_original'])][1])**2
                ) * 111,  # Rough conversion to km
                axis=1
            )
        
        # Parameter-based filtering
        st.header("Parameter-Based Analysis")
        
        # Calculate density with progress
        with st.spinner("Calculating density distribution..."):
            kde = gaussian_kde(x.T)
            data['density'] = kde(x.T)
        
        # Filter based on parameters
        density_cutoff = data['density'].quantile(density_threshold/100)
        distance_cutoff = data['distance_original'].quantile(percentile_coverage/100)
        
        filtered_data = data[
            (data['density'] >= density_cutoff) & 
            (data['distance_original'] <= distance_cutoff)
        ].copy()
        
        st.info(f"**Filtered Data:** {len(filtered_data)} orders ({len(filtered_data)/len(data)*100:.1f}% of total)")
        
        # Clustering on filtered data
        if len(filtered_data) > n_clusters:
            with st.spinner("Performing filtered clustering..."):
                x_filtered = filtered_data[['Lat', 'Long']].values
                kmeans_filtered, y_kmeans_filtered = perform_clustering(x_filtered, n_clusters)
                filtered_data['cluster_filtered'] = y_kmeans_filtered
                
                # Calculate distances for filtered clustering
                filtered_data['distance_filtered'] = filtered_data.apply(
                    lambda row: np.sqrt(
                        (row['Lat'] - kmeans_filtered.cluster_centers_[int(row['cluster_filtered'])][0])**2 +
                        (row['Long'] - kmeans_filtered.cluster_centers_[int(row['cluster_filtered'])][1])**2
                    ) * 111,
                    axis=1
                )
        
        # Visualization
        st.header("Clustering Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Clustering (100% Coverage)")
            fig_original = px.scatter(
                data, x='Long', y='Lat', color='cluster_original',
                title=f"Original: {n_clusters} Clusters, {len(data)} Orders"
            )
            # Add cluster centers
            centers_original = kmeans_original.cluster_centers_
            fig_original.add_trace(go.Scatter(
                x=centers_original[:, 1],
                y=centers_original[:, 0],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='Cluster Centers'
            ))
            st.plotly_chart(fig_original, use_container_width=True)
        
        with col2:
            if len(filtered_data) > n_clusters:
                st.subheader(f"Filtered Clustering ({len(filtered_data)} Orders)")
                fig_filtered = px.scatter(
                    filtered_data, x='Long', y='Lat', color='cluster_filtered',
                    title=f"Filtered: {n_clusters} Clusters, {len(filtered_data)} Orders"
                )
                # Add cluster centers
                centers_filtered = kmeans_filtered.cluster_centers_
                fig_filtered.add_trace(go.Scatter(
                    x=centers_filtered[:, 1],
                    y=centers_filtered[:, 0],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='x'),
                    name='Cluster Centers'
                ))
                st.plotly_chart(fig_filtered, use_container_width=True)
            else:
                st.warning("Insufficient data for clustering after filtering")
        
        # Interactive Map using Plotly
        st.header("Interactive Map")
        
        # Create map using Plotly
        fig_map = go.Figure()
        
        # Colors for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
        
        # Add original cluster centers
        centers_original = kmeans_original.cluster_centers_
        fig_map.add_trace(go.Scatter(
            x=centers_original[:, 1],
            y=centers_original[:, 0],
            mode='markers',
            marker=dict(size=20, color='red', symbol='x'),
            name='Original Cluster Centers',
            text=[f'Cluster {i}' for i in range(n_clusters)],
            hovertemplate='<b>%{text}</b><br>Lat: %{y:.4f}<br>Long: %{x:.4f}<extra></extra>'
        ))
        
        # Add filtered cluster centers if available
        if len(filtered_data) > n_clusters:
            centers_filtered = kmeans_filtered.cluster_centers_
            fig_map.add_trace(go.Scatter(
                x=centers_filtered[:, 1],
                y=centers_filtered[:, 0],
                mode='markers',
                marker=dict(size=20, color='orange', symbol='star'),
                name='Filtered Cluster Centers',
                text=[f'Filtered Cluster {i}' for i in range(n_clusters)],
                hovertemplate='<b>%{text}</b><br>Lat: %{y:.4f}<br>Long: %{x:.4f}<extra></extra>'
            ))
        
        # Add order points (optimized for performance)
        sample_size = min(1000, len(data))  # Reduced sample size for faster rendering
        sample_data = data.sample(n=sample_size, random_state=42)
        
        for cluster_id in range(n_clusters):
            cluster_data = sample_data[sample_data['cluster_original'] == cluster_id]
            if len(cluster_data) > 0:
                fig_map.add_trace(go.Scatter(
                    x=cluster_data['Long'],
                    y=cluster_data['Lat'],
                    mode='markers',
                    marker=dict(size=2, color=colors[cluster_id % len(colors)]),  # Smaller markers
                    name=f'Cluster {cluster_id} Orders',
                    hovertemplate='<b>Order</b><br>Lat: %{y:.4f}<br>Long: %{x:.4f}<extra></extra>',
                    showlegend=False
                ))
        
        # Update layout for map
        fig_map.update_layout(
            title="Interactive Map - Clusters and Centers",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500,  # Reduced height
            hovermode='closest'
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Cost Analysis
        st.header("Cost Analysis")
        
        # Cost parameters
        cost_per_km = 10
        fixed_store_cost = 500000
        
        # Calculate costs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Data (100% Coverage)")
            total_distance_original = data['distance_original'].sum()
            total_cost_original = total_distance_original * cost_per_km + (fixed_store_cost * n_clusters)
            
            st.metric("Total Distance", f"{total_distance_original:,.0f} km")
            st.metric("Delivery Cost", f"₹{total_distance_original * cost_per_km:,.0f}")
            st.metric("Store Cost", f"₹{fixed_store_cost * n_clusters:,.0f}")
            st.metric("Total Cost", f"₹{total_cost_original:,.0f}")
        
        with col2:
            if len(filtered_data) > n_clusters:
                st.subheader(f"Filtered Data ({len(filtered_data)/len(data)*100:.1f}% Coverage)")
                total_distance_filtered = filtered_data['distance_filtered'].sum()
                total_cost_filtered = total_distance_filtered * cost_per_km + (fixed_store_cost * n_clusters)
                
                st.metric("Total Distance", f"{total_distance_filtered:,.0f} km")
                st.metric("Delivery Cost", f"₹{total_distance_filtered * cost_per_km:,.0f}")
                st.metric("Store Cost", f"₹{fixed_store_cost * n_clusters:,.0f}")
                st.metric("Total Cost", f"₹{total_cost_filtered:,.0f}")
            else:
                st.warning("Insufficient data for cost analysis")
        
        # Cost comparison
        if len(filtered_data) > n_clusters:
            st.subheader("Cost Comparison")
            
            cost_savings = total_cost_original - total_cost_filtered
            savings_percentage = (cost_savings / total_cost_original) * 100
            
            if cost_savings > 0:
                st.success(f"**Cost Savings: ₹{cost_savings:,.0f} ({savings_percentage:.1f}%)**")
            else:
                st.warning(f"**Additional Cost: ₹{abs(cost_savings):,.0f}**")
            
            # Comparison chart
            comparison_data = pd.DataFrame({
                'Metric': ['Total Cost', 'Delivery Cost', 'Store Cost'],
                'Original': [total_cost_original, total_distance_original * cost_per_km, fixed_store_cost * n_clusters],
                'Filtered': [total_cost_filtered, total_distance_filtered * cost_per_km, fixed_store_cost * n_clusters]
            })
            
            fig_comparison = px.bar(
                comparison_data, x='Metric', y=['Original', 'Filtered'],
                title="Cost Comparison: Original vs Filtered",
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Download results
        st.header("Download Results")
        
        if len(filtered_data) > n_clusters:
            # Prepare data for download
            results_data = data.copy()
            results_data['cluster_center_lat'] = results_data['cluster_original'].map(
                {i: kmeans_original.cluster_centers_[i][0] for i in range(n_clusters)}
            )
            results_data['cluster_center_long'] = results_data['cluster_original'].map(
                {i: kmeans_original.cluster_centers_[i][1] for i in range(n_clusters)}
            )
            
            csv = results_data.to_csv(index=True)
            st.download_button(
                label="Download Analysis Results (CSV)",
                data=csv,
                file_name="clustering_analysis_results.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure your CSV has 'Lat' and 'Long' columns")

else:
    st.info("Please upload a CSV file to begin the analysis")
    
    st.markdown("""
    ### Expected CSV Format:
    ```
    Order ID,Lat,Long
    1,12.925978,77.615465
    2,12.902758,77.593012
    3,12.921465,77.734314
    ```
    """)

# Performance tips
st.sidebar.markdown("---")
st.sidebar.markdown("** Performance Tips:**")
st.sidebar.markdown("- Disable elbow method for faster loading")
st.sidebar.markdown("- Reduce max clusters for elbow method")
st.sidebar.markdown("- Use smaller datasets for better performance")

