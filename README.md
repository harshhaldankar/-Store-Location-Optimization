# 🏪 Store Location Optimization

A comprehensive Streamlit application for analyzing order coordinates and optimizing store locations to reduce delivery costs through intelligent clustering analysis.

## 🚀 Features

- **Interactive Data Upload**: Upload CSV files with order coordinates
- **Smart Clustering**: K-means clustering with elbow method optimization
- **Cost Analysis**: Comprehensive cost comparison between single and multi-store approaches
- **Density Analysis**: Kernel density estimation for order distribution analysis
- **Peripheral Analysis**: Identification of low-density, high-distance areas
- **Downloadable Results**: Export analysis results as CSV

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Upload your CSV file** with the following format:
   ```
   Order ID,Lat,Long
   1,12.925978,77.615465
   2,12.902758,77.593012
   3,12.921465,77.734314
   ```

2. **Configure parameters** in the sidebar:
   - Cost per km
   - Fixed store cost
   - Number of clusters
   - Random state

3. **Explore the analysis results:**
   - Data overview and statistics
   - Clustering analysis with elbow method
   - Interactive maps
   - Cost analysis and comparisons
   - Density and peripheral analysis
   - Strategic recommendations

## 📊 Analysis Components

### 1. **Data Overview**
- Total order count
- Geographic boundaries
- Data quality checks

### 2. **Clustering Analysis**
- Elbow method for optimal cluster count
- K-means clustering results
- Cluster statistics and centroids

### 3. **Interactive Mapping**
- Folium-based interactive maps
- Color-coded clusters
- Cluster center markers
- Order location visualization

### 4. **Cost Analysis**
- Distance calculations using Haversine formula
- Delivery cost calculations
- Store cost considerations
- Single vs. multi-store cost comparison

### 5. **Density Analysis**
- Kernel density estimation
- Order distribution patterns
- Peripheral area identification

### 6. **Strategic Insights**
- Store location recommendations
- Cost optimization strategies
- Operational considerations
- Future planning guidance

### Clustering Parameters
- **Number of clusters**: How many store locations to consider
- **Random state**: For reproducible clustering results

## 📁 File Structure

```
Cost_cutting Analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── Assesment_2.ipynb     # Original Jupyter notebook
└── Order coordinates.csv  # Sample data file
```

##  Use Cases

- **E-commerce delivery optimization**
- **Retail store location planning**
- **Logistics and supply chain optimization**
- **Service area planning**
- **Cost reduction analysis**

##  Key Benefits

- **Cost Reduction**: Identify optimal store locations to minimize delivery costs
- **Data-Driven Decisions**: Make informed decisions based on order patterns
- **Interactive Analysis**: Explore data through interactive visualizations
- **Scalable Solution**: Handle large datasets efficiently
- **Professional Output**: Generate reports and insights for stakeholders

##  Technical Details

- **Clustering Algorithm**: K-means with elbow method optimization
- **Distance Calculation**: Haversine formula for geographic distances
- **Visualization**: Plotly for charts, Folium for maps
- **Data Processing**: Pandas for data manipulation and analysis
- **Machine Learning**: Scikit-learn for clustering algorithms


