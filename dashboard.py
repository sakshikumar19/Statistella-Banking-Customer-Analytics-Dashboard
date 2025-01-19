# Here is the code for the dashboard:

# 🔗 LINK TO DEPLOYED DASHBOARD: https://statistella-db.streamlit.app/ 🌐

# You can view the deployed dashboard from the above link! You don't need to run the code for it.

# Here is the associated GitHub repository: https://github.com/sakshikumar19/Statistella-Banking-Customer-Analytics-Dashboard 📂

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import folium
from folium.plugins import MarkerCluster
from scipy import stats

# Set page config
st.set_page_config(page_title="Banking Customer Analytics", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .plot-container {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Banking Customer Analytics Dashboard")
st.markdown("Comprehensive analysis of customer behavior, demographics, and performance metrics")


def load_data():
    df = pd.read_csv('Customer-Churn-Records.csv')
    
    # Calculate additional metrics
    df['SalaryUsedPercentage'] = (df['Balance'] / df['EstimatedSalary']) * 100
    
    # More robust value segmentation with proper handling of edge cases
    def create_value_segments(df):
        if df['Balance'].nunique() <= 3:
            # If there are very few unique values, use simple categorization
            df['ValueSegment'] = pd.Categorical(
                pd.qcut(df['Balance'].rank(method='first'), 
                       q=3, 
                       labels=['Low', 'Medium', 'High'])
            )
        else:
            try:
                # Handle cases with many ties by using rank
                df['ValueSegment'] = pd.Categorical(
                    pd.qcut(df['Balance'].rank(method='first'),
                           q=3,
                           labels=['Low', 'Medium', 'High'])
                )
            except Exception as e:
                st.warning(f"Warning: Using alternative segmentation method due to: {str(e)}")
                # Fallback to simple percentile-based cutoffs
                percentiles = df['Balance'].quantile([0, 0.33, 0.67, 1.0]).values
                df['ValueSegment'] = pd.cut(
                    df['Balance'],
                    bins=percentiles,
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                )
        return df

    # Apply the segmentation
    df = create_value_segments(df)
    
    # Age grouping
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 30, 50, 100], 
                           labels=['Young', 'Middle-aged', 'Senior'],
                           include_lowest=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

geography_filter = st.sidebar.multiselect(
    "Geography", options=sorted(df['Geography'].unique()),
    default=sorted(df['Geography'].unique())
)

age_range = st.sidebar.slider(
    "Age Range", 
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

card_type_filter = st.sidebar.multiselect(
    "Card Type", options=sorted(df['Card Type'].unique()),
    default=sorted(df['Card Type'].unique())
)

value_segment_filter = st.sidebar.multiselect(
    "Value Segment", options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

# Apply filters
filtered_df = df[
    (df['Geography'].isin(geography_filter)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Card Type'].isin(card_type_filter)) &
    (df['ValueSegment'].isin(value_segment_filter))
]

# Main dashboard tabs
tabs = st.tabs([
    "Customer Segmentation", 
    "Churn Analysis",
    "Customer Satisfaction",
    "Transaction Patterns",
    "Retention Analysis",
    "Trend Analysis",
    "High-Value Analysis"
    ])

# Tab 1: Customer Segmentation
with tabs[0]:
    st.header("Customer Segmentation Analysis")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{len(filtered_df):,}",
            f"{(len(filtered_df)/len(df)*100)-100:.1f}%"
        )
    
    with col2:
        avg_balance = filtered_df['Balance'].mean()
        st.metric(
            "Average Balance",
            f"${avg_balance:,.2f}",
            f"{(avg_balance/df['Balance'].mean()*100)-100:.1f}%"
        )
    
    with col3:
        avg_products = filtered_df['NumOfProducts'].mean()
        st.metric(
            "Avg Products/Customer",
            f"{avg_products:.2f}",
            f"{(avg_products/df['NumOfProducts'].mean()*100)-100:.1f}%"
        )
    
    with col4:
        active_rate = filtered_df['IsActiveMember'].mean() * 100
        st.metric(
            "Active Customer Rate",
            f"{active_rate:.1f}%",
            f"{active_rate - (df['IsActiveMember'].mean()*100):.1f}%"
        )

    st.subheader("Demographic Segmentation")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution by Gender
        fig_age = px.histogram(
            filtered_df,
            x='Age',
            color='Gender',
            title='Age Distribution by Gender',
            nbins=30,
            color_discrete_sequence=['#FF69B4', '#4169E1']
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Geography and Card Type Distribution
        fig_geo = px.sunburst(
            filtered_df,
            path=['Geography', 'Card Type'],
            title='Customer Distribution by Geography and Card Type'
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    st.subheader("Behavioral Segmentation")
    col1, col2 = st.columns(2)
    
    with col1:
        # Product Usage by Value Segment
        product_usage = filtered_df.groupby('ValueSegment')['NumOfProducts'].value_counts().unstack()
        fig_products = px.bar(
            product_usage,
            title='Product Usage by Value Segment',
            barmode='group'
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    with col2:
        # Points Distribution by Activity Status
        fig_points = px.box(
            filtered_df,
            x='IsActiveMember',
            y='Point Earned',
            color='ValueSegment',
            title='Points Distribution by Activity Status and Value Segment'
        )
        st.plotly_chart(fig_points, use_container_width=True)

    st.subheader("Top Performers Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 5 by Balance
        top_balance = filtered_df.nlargest(5, 'Balance')[['Surname', 'Balance', 'Geography']]
        st.write("Top 5 Customers by Balance")
        st.dataframe(top_balance)
    
    with col2:
        # Top 5 by Points
        top_points = filtered_df.nlargest(5, 'Point Earned')[['Surname', 'Point Earned', 'Geography']]
        st.write("Top 5 Customers by Points Earned")
        st.dataframe(top_points)
    
    # Add Regional Performance
    st.subheader("Regional Performance")
    
    regional_metrics = filtered_df.groupby('Geography').agg({
        'Balance': 'mean',
        'Satisfaction Score': 'mean',
        'Point Earned': 'mean',
        'NumOfProducts': 'mean'
    }).round(2)
    
    st.dataframe(regional_metrics.style.highlight_max(axis=0))

    # Map Integration in the Tab
    st.subheader("Customer Geography Distribution")
    
    # Coordinates mapping for geography
    location_coordinates = {
        'France': (46.603354, 1.888334),
        'Germany': (51.165691, 10.451526),
        'Spain': (40.463667, -3.74922),
        'Italy': (41.87194, 12.56738),
        'Portugal': (39.399872, -8.224454),
    }
    df['Coordinates'] = df['Geography'].map(location_coordinates)

    # Folium map setup
    m = folium.Map(location=[51.165691, 10.451526], zoom_start=4, tiles='cartodb positron')
    marker_cluster = MarkerCluster().add_to(m)

    # Adding markers
    for idx, row in df.dropna(subset=['Coordinates']).iterrows():
        lat, lon = row['Coordinates']
        folium.Marker(location=[lat, lon], popup=row['Surname']).add_to(marker_cluster)

    # Save map to HTML and display in Streamlit
    m.save("customer_geography_distribution.html")
    st.components.v1.html(open("customer_geography_distribution.html", "r").read(), height=600)

# Tab 2: Churn Analysis
with tabs[1]:
    st.header("Churn Analysis")
    
    # Churn Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churn_rate = filtered_df['Exited'].mean() * 100
        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.1f}%",
            f"{churn_rate - (df['Exited'].mean()*100):.1f}%"
        )
    
    with col2:
        active_churn = filtered_df[filtered_df['IsActiveMember']==1]['Exited'].mean() * 100
        st.metric(
            "Active Customer Churn",
            f"{active_churn:.1f}%"
        )
    
    with col3:
        inactive_churn = filtered_df[filtered_df['IsActiveMember']==0]['Exited'].mean() * 100
        st.metric(
            "Inactive Customer Churn",
            f"{inactive_churn:.1f}%"
        )

    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by Demographics
        churn_demo = filtered_df.groupby(['Geography', 'Gender'])['Exited'].mean().reset_index()
        fig_churn_demo = px.bar(
            churn_demo,
            x='Geography',
            y='Exited',
            color='Gender',
            title='Churn Rate by Geography and Gender',
            barmode='group',
            color_discrete_sequence=['#FF69B4', '#4169E1']
        )
        st.plotly_chart(fig_churn_demo, use_container_width=True)
    
    with col2:
        # Churn by Value Segment and Products
        churn_value = filtered_df.groupby(['ValueSegment', 'NumOfProducts'])['Exited'].mean().reset_index()
        fig_churn_value = px.bar(
            churn_value,
            x='ValueSegment',
            y='Exited',
            color='NumOfProducts',
            title='Churn Rate by Value Segment and Number of Products',
            barmode='group'
        )
        st.plotly_chart(fig_churn_value, use_container_width=True)

    # Churn Factor Analysis
    st.subheader("Churn Factor Analysis")
    
    # Calculate statistical differences
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'Point Earned']
    stats_df = pd.DataFrame(columns=['Feature', 'Churned_Mean', 'Non_Churned_Mean', 'Difference', 'P_Value'])
    
    for col in numeric_cols:
        churned = filtered_df[filtered_df['Exited']==1][col]
        non_churned = filtered_df[filtered_df['Exited']==0][col]
        t_stat, p_val = stats.ttest_ind(churned, non_churned)
        
        stats_df = pd.concat([stats_df, pd.DataFrame({
            'Feature': [col],
            'Churned_Mean': [churned.mean()],
            'Non_Churned_Mean': [non_churned.mean()],
            'Difference': [churned.mean() - non_churned.mean()],
            'P_Value': [p_val]
        })])
    
    st.dataframe(stats_df.round(4), use_container_width=True)

# Tab 3: Customer Satisfaction
with tabs[2]:
    st.header("Customer Satisfaction Analysis")
    
    # Satisfaction Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_satisfaction = filtered_df['Satisfaction Score'].mean()
        st.metric(
            "Average Satisfaction",
            f"{avg_satisfaction:.2f}",
            f"{(avg_satisfaction/df['Satisfaction Score'].mean()*100)-100:.1f}%"
        )
    
    with col2:
        complaint_rate = filtered_df['Complain'].mean() * 100
        st.metric(
            "Complaint Rate",
            f"{complaint_rate:.1f}%"
        )
    
    with col3:
        satisfied_retention = filtered_df[filtered_df['Satisfaction Score'] >= 4]['Exited'].mean() * 100
        st.metric(
            "Satisfied Customer Retention",
            f"{100 - satisfied_retention:.1f}%"
        )

    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction by Product Usage
        fig_satisfaction = px.box(
            filtered_df,
            x='NumOfProducts',
            y='Satisfaction Score',
            color='IsActiveMember',
            title='Satisfaction by Product Usage and Activity Status'
        )
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        # Complaints Analysis
        complaints_analysis = filtered_df.groupby(['Geography', 'Complain'])['Satisfaction Score'].mean().reset_index()
        fig_complaints = px.bar(
            complaints_analysis,
            x='Geography',
            y='Satisfaction Score',
            color='Complain',
            title='Average Satisfaction Score by Geography and Complaints',
            barmode='group'
        )
        st.plotly_chart(fig_complaints, use_container_width=True)

    # Satisfaction Driver Analysis
    st.subheader("Satisfaction Drivers")

    # Calculate correlations with satisfaction
    satisfaction_corr = filtered_df[['Satisfaction Score', 'CreditScore', 'Age', 'Tenure', 
                                    'Balance', 'NumOfProducts', 'Point Earned']].corr()['Satisfaction Score']

    # Drop the self-correlation (Satisfaction Score with itself)
    satisfaction_corr = satisfaction_corr.drop('Satisfaction Score')

    # Plot the correlations
    fig_corr = go.Figure(go.Bar(
        x=satisfaction_corr.index,
        y=satisfaction_corr.values,
        marker_color='#1f77b4'
    ))
    fig_corr.update_layout(
        title='Correlation with Satisfaction Score',
        xaxis_title='Features',
        yaxis_title='Correlation Coefficient'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Satisfaction Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction by Demographics
        satisfaction_demo = filtered_df.groupby(['Geography', 'Gender'])['Satisfaction Score'].mean().reset_index()
        fig_satisfaction_demo = px.bar(
            satisfaction_demo,
            x='Geography',
            y='Satisfaction Score',
            color='Gender',
            title='Satisfaction Score by Geography and Gender',
            barmode='group'
        )
        st.plotly_chart(fig_satisfaction_demo, use_container_width=True)
    
    with col2:
        # Satisfaction by Card Type
        satisfaction_card = filtered_df.groupby('Card Type')['Satisfaction Score'].mean().reset_index()
        fig_satisfaction_card = px.bar(
            satisfaction_card,
            x='Card Type',
            y='Satisfaction Score',
            title='Average Satisfaction Score by Card Type',
            color='Card Type'
        )
        st.plotly_chart(fig_satisfaction_card, use_container_width=True)
    
    # Add complaints analysis section
    st.subheader("Complaints Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for complaints by geography
        complaints_geo = filtered_df.groupby('Geography')['Complain'].mean().reset_index()
        fig_complaints_geo = px.bar(
            complaints_geo,
            x='Geography',
            y='Complain',
            title='Complaint Rate by Geography',
            color='Geography'
        )
        fig_complaints_geo.update_layout(yaxis_title='Complaint Rate (%)')
        st.plotly_chart(fig_complaints_geo, use_container_width=True)
    
    with col2:
        # Complaints by card type
        complaints_card = filtered_df.groupby('Card Type')['Complain'].mean().reset_index()
        fig_complaints_card = px.bar(
            complaints_card,
            x='Card Type',
            y='Complain',
            title='Complaint Rate by Card Type',
            color='Card Type'
        )
        fig_complaints_card.update_layout(yaxis_title='Complaint Rate (%)')
        st.plotly_chart(fig_complaints_card, use_container_width=True)

# Tab 4: Transaction Patterns
with tabs[3]:
    st.header("Transaction Pattern Analysis")
    
    # Transaction Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_balance = filtered_df['Balance'].mean()
        st.metric(
            "Average Balance",
            f"${avg_balance:,.2f}"
        )
    
    with col2:
        avg_salary_usage = filtered_df['SalaryUsedPercentage'].mean()
        st.metric(
            "Avg Salary Usage",
            f"{avg_salary_usage:.1f}%"
        )
    
    with col3:
        high_spenders = (filtered_df['SalaryUsedPercentage'] > 50).mean() * 100
        st.metric(
            "High Spenders",
            f"{high_spenders:.1f}%"
        )

    col1, col2 = st.columns(2)
    
    with col1:
        # Balance Distribution by Card Type
        fig_balance = px.violin(
            filtered_df,
            x='Card Type',
            y='Balance',
            color='IsActiveMember',
            title='Balance Distribution by Card Type and Activity Status',
            box=True
        )
        st.plotly_chart(fig_balance, use_container_width=True)
    
# Tab 4: Transaction Patterns (continued)
    with col2:
        # Salary Usage Analysis
        fig_salary = px.histogram(
            filtered_df,
            x='SalaryUsedPercentage',
            color='ValueSegment',
            title='Salary Usage Distribution by Value Segment',
            nbins=30
        )
        st.plotly_chart(fig_salary, use_container_width=True)

    # Product Usage Analysis
    st.subheader("Product Usage Analysis")
    
    # Product distribution by segment
    product_dist = pd.crosstab(
        filtered_df['ValueSegment'],
        filtered_df['NumOfProducts']
    ).apply(lambda x: x/x.sum() * 100, axis=1)
    
    fig_products = px.bar(
        product_dist,
        title='Product Distribution by Value Segment (%)',
        barmode='group'
    )
    st.plotly_chart(fig_products, use_container_width=True)

# Tab 5: Retention Analysis
with tabs[4]:
    st.header("Retention Analysis")
    
    # Retention Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retention_rate = (1 - filtered_df['Exited'].mean()) * 100
        st.metric(
            "Overall Retention Rate",
            f"{retention_rate:.1f}%"
        )
    
    with col2:
        avg_tenure = filtered_df['Tenure'].mean()
        st.metric(
            "Average Tenure (months)",
            f"{avg_tenure:.1f}"
        )
    
    with col3:
        loyal_customers = (filtered_df['Tenure'] > 36).mean() * 100
        st.metric(
            "Long-term Customers",
            f"{loyal_customers:.1f}%"
        )

    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure Analysis
        tenure_metrics = filtered_df.groupby('Tenure').agg({
            'Exited': 'mean',
            'Satisfaction Score': 'mean',
            'CustomerId': 'count'
        }).reset_index()
        
        fig_tenure = go.Figure()
        fig_tenure.add_trace(go.Bar(
            x=tenure_metrics['Tenure'],
            y=tenure_metrics['CustomerId'],
            name='Customer Count',
            yaxis='y2'
        ))
        fig_tenure.add_trace(go.Scatter(
            x=tenure_metrics['Tenure'],
            y=tenure_metrics['Satisfaction Score'],
            name='Satisfaction',
            line=dict(color='green')
        ))
        
        fig_tenure.update_layout(
            title='Customer Count and Satisfaction by Tenure',
            yaxis=dict(title='Satisfaction Score'),
            yaxis2=dict(title='Customer Count', overlaying='y', side='right')
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with col2:
        # Activity Impact
        activity_metrics = filtered_df.groupby(['IsActiveMember', 'ValueSegment'])['Exited'].mean().reset_index()
        fig_activity = px.bar(
            activity_metrics,
            x='ValueSegment',
            y='Exited',
            color='IsActiveMember',
            title='Churn Rate by Activity Status and Value Segment',
            barmode='group'
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    # Rewards Analysis
    st.subheader("Rewards Impact Analysis")
    
    rewards_analysis = pd.qcut(filtered_df['Point Earned'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    rewards_metrics = filtered_df.groupby(rewards_analysis).agg({
        'Exited': 'mean',
        'Satisfaction Score': 'mean',
        'CustomerId': 'count'
    }).round(3)
    rewards_metrics = rewards_metrics.rename(columns={'CustomerId': 'Customer Count'})
    st.dataframe(rewards_metrics.reset_index(), use_container_width=True)

# Tab 6: Trend Analysis
with tabs[5]:
    st.header("Trend Analysis")
    
    # Age Group Analysis
    st.subheader("Age Group Trends")
    
    age_metrics = filtered_df.groupby('AgeGroup').agg({
        'Exited': 'mean',
        'Satisfaction Score': 'mean',
        'Balance': 'mean',
        'Point Earned': 'mean',
        'CustomerId': 'count'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Group Metrics Visualization
        fig_age_metrics = go.Figure()
        
        for col in ['Satisfaction Score', 'Exited']:
            fig_age_metrics.add_trace(go.Bar(
                name=col,
                x=age_metrics.index,
                y=age_metrics[col]
            ))
        
        fig_age_metrics.update_layout(
            title='Key Metrics by Age Group',
            barmode='group'
        )
        st.plotly_chart(fig_age_metrics, use_container_width=True)
    
    with col2:
        # Regional Trends
        geography_metrics = filtered_df.groupby('Geography').agg({
            'Exited': 'mean',
            'Satisfaction Score': 'mean',
            'Balance': 'mean'
        }).round(2)
        
        fig_geo_metrics = px.bar(
            geography_metrics,
            barmode='group',
            title='Regional Performance Metrics'
        )
        st.plotly_chart(fig_geo_metrics, use_container_width=True)

    # Cohort Analysis
    st.subheader("Tenure-based Cohort Analysis")
    
    tenure_cohorts = pd.qcut(filtered_df['Tenure'], q=4, labels=['New', 'Developing', 'Established', 'Loyal'])
    cohort_metrics = filtered_df.groupby(tenure_cohorts).agg({
        'Exited': 'mean',
        'Satisfaction Score': 'mean',
        'Balance': 'mean',
        'Point Earned': 'mean',
        'CustomerId': 'count'
    }).round(2)

    cohort_metrics = cohort_metrics.rename(columns={'CustomerId': 'Customer Count'})

    st.dataframe(cohort_metrics.reset_index(), use_container_width=True)

with tabs[6]:
    st.header("Customer Value Analysis")

    # Define high-value customers
    filtered_df['High_Value_Score'] = (
        (filtered_df['Balance'] > filtered_df['Balance'].quantile(0.75)) * 0.4 +
        (filtered_df['NumOfProducts'] > 1) * 0.3 +
        (filtered_df['Satisfaction Score'] > 4) * 0.3
    )
    filtered_df['Is_High_Value'] = filtered_df['High_Value_Score'] > 0.7

    # Define low-value customers
    filtered_df['Low_Value_Score'] = (
        (filtered_df['Balance'] < filtered_df['Balance'].quantile(0.50)) * 0.5 +
        (filtered_df['NumOfProducts'] == 1) * 0.3 +
        (filtered_df['Satisfaction Score'] <= 3) * 0.2
    )
    filtered_df['Is_Low_Value'] = filtered_df['Low_Value_Score'] > 0.7

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        high_value_pct = filtered_df['Is_High_Value'].mean() * 100
        st.metric("High-Value Customers", f"{high_value_pct:.1f}%")

    with col2:
        low_value_pct = filtered_df['Is_Low_Value'].mean() * 100
        st.metric("Low-Value Customers", f"{low_value_pct:.1f}%")

    with col3:
        potential_hv_pct = ((filtered_df['Low_Value_Score'] < 0.7) & (filtered_df['High_Value_Score'] < 0.7)).mean() * 100
        st.metric("Potential High-Value Customers", f"{potential_hv_pct:.1f}%")

    # High-Value Customer Profile
    st.subheader("High-Value Customer Profile")
    fig_high_value = px.histogram(
        filtered_df[filtered_df['Is_High_Value']],
        x='Age',
        color='Gender',
        title='Age Distribution of High-Value Customers',
        nbins=20,
        color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'}  # Distinct colors for Gender
    )
    st.plotly_chart(fig_high_value, use_container_width=True)

    # Low-Value Customer Profile
    st.subheader("Low-Value Customer Profile")
    fig_low_value = px.histogram(
        filtered_df[filtered_df['Is_Low_Value']],
        x='Age',
        color='Gender',
        title='Age Distribution of Low-Value Customers',
        nbins=20,
        color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'}  # Distinct colors for Gender
    )
    st.plotly_chart(fig_low_value, use_container_width=True)

    # Geographical Distribution with Grouped Bar Chart
    st.subheader("Geographical Distribution by Customer Value Segment")

    # Aggregating data for a grouped bar chart (segmented by Geography and Value Segment)
    geo_dist = filtered_df.groupby(['Geography', 'Is_High_Value', 'Is_Low_Value']).size().reset_index(name='count')

    # Converting True/False to readable values
    geo_dist['Value Segment'] = geo_dist['Is_High_Value'].map({True: 'High Value', False: 'Low Value'})

    # Create a grouped bar chart
    fig_geo_bar = px.bar(
        geo_dist,
        x='Geography',
        y='count',
        color='Value Segment',
        title='Geographical Distribution by Customer Value Segment',
        labels={"count": "Number of Customers", "Geography": "Region", "Value Segment": "Customer Value Segment"},
        barmode='group',
        color_discrete_map={'High Value': 'orange', 'Low Value': 'darkred'}  # Different colors for segments
    )

    # Styling the plot for a better appearance
    fig_geo_bar.update_layout(
        plot_bgcolor='white',
        title_x=0.5,
        xaxis_title='Geography',
        yaxis_title='Number of Customers',
        legend_title='Customer Value Segment'
    )

    # Display the grouped bar chart
    st.plotly_chart(fig_geo_bar, use_container_width=True)

    # Behavioral Insights
    st.subheader("Behavioral Insights")
    behavior_metrics = filtered_df.groupby(['Is_High_Value', 'Is_Low_Value']).agg({
        'Balance': 'mean',
        'Satisfaction Score': 'mean',
        'Point Earned': 'mean',
        'Tenure': 'mean',
        'CustomerId': 'count'
    }).round(2)
    st.dataframe(behavior_metrics.style.highlight_max(axis=0), use_container_width=True)

    # Potential High-Value Customers
    st.subheader("Potential High-Value Customer Analysis")
    potential_hv = filtered_df[(filtered_df['Low_Value_Score'] < 0.7) & (filtered_df['High_Value_Score'] < 0.7)]
    fig_potential_hv = px.scatter(
        potential_hv,
        x='Balance',
        y='Satisfaction Score',
        color='Geography',
        title='Potential High-Value Customer Distribution',
        hover_data=['NumOfProducts', 'Tenure'],
        color_discrete_map={'East': 'cyan', 'West': 'magenta'}  # Different colors for Geography
    )
    st.plotly_chart(fig_potential_hv, use_container_width=True)

    # Recommendations
    st.subheader("Strategic Recommendations")
    st.markdown("""
    ### For High-Value Customers
    - Enhance rewards programs
    - Offer exclusive premium services
    - Retain loyalty through personal outreach

    ### For Low-Value Customers
    - Conduct financial literacy workshops
    - Provide targeted promotions for increased engagement
    - Simplify onboarding for additional products

    ### For Potential High-Value Customers
    - Use predictive models to identify upgrade likelihood
    - Focus on improving satisfaction scores through surveys and feedback
    - Offer limited-time benefits to encourage increased usage
    """)

# Download section
st.header("Export Data")
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Download Filtered Data",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_banking_data.csv',
        mime='text/csv'
    )

with col2:
    st.download_button(
        label="Download Cohort Analysis",
        data=cohort_metrics.to_csv().encode('utf-8'),
        file_name='cohort_analysis.csv',
        mime='text/csv'
    )
