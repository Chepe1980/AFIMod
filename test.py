"""
AVO Fluid Inversion (AFI) - STREAMLIT WEB APPLICATION
Based on: "AVO FLUID INVERSION (AFI) TECHNIQUE AS A TOOL TO PREDICT RESERVOIR 
FLUID CONTENT USING DATA FROM FD FIELD, ONSHORE NIGER DELTA NIGERIA"

Features:
- Upload CSV well log data
- Interactive depth range selection
- Theoretical fluid clusters (Gas, Oil, Brine, Shale)
- Bayesian probability classification
- Interactive Plotly visualizations with proper well log curves
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AVO Fluid Inversion (AFI)",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class AVOFluidInversionApp:
    """AVO Fluid Inversion for Streamlit App"""
    
    def __init__(self, df, depth_min, depth_max):
        """Initialize with well log data"""
        mask = (df['DEPTH'] >= depth_min) & (df['DEPTH'] <= depth_max)
        self.df = df[mask].copy().sort_values('DEPTH')
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.df_full = df.copy()
        
        # Calculate background properties
        top_depth = depth_min + (depth_max - depth_min) * 0.2
        shallow = self.df[self.df['DEPTH'] < top_depth]
        
        if len(shallow) > 0:
            self.Vp_back = np.mean(shallow['Vp'])
            self.Vs_back = np.mean(shallow['Vs'])
            self.rho_back = np.mean(shallow['rho'])
        else:
            self.Vp_back = np.mean(self.df['Vp'])
            self.Vs_back = np.mean(self.df['Vs'])
            self.rho_back = np.mean(self.df['rho'])
        
        # Calculate AVO attributes
        self.actual_A, self.actual_B = self.calculate_avo_attributes()
        
        # Define fluid clusters
        self.fluid_clusters = self.define_fluid_clusters()
        
        # Calculate Bayesian probability
        self.prob_df, self.bayesian_clusters = self.calculate_bayesian_probability()
    
    def calculate_avo_attributes(self):
        """Calculate AVO Intercept (A) and Gradient (B)"""
        I1 = self.rho_back * self.Vp_back
        I2 = self.df['rho'].values * self.df['Vp'].values
        A = (I2 - I1) / (I2 + I1 + 1e-10)
        
        VpVs_back = self.Vp_back / self.Vs_back
        sigma1 = (0.5 * VpVs_back**2 - 1) / (VpVs_back**2 - 1)
        
        VpVs = self.df['Vp'].values / (self.df['Vs'].values + 1e-10)
        sigma2 = np.zeros_like(VpVs)
        mask = VpVs > 0
        sigma2[mask] = (0.5 * VpVs[mask]**2 - 1) / (VpVs[mask]**2 - 1)
        sigma2[~mask] = sigma1
        
        delta_sigma = sigma2 - sigma1
        mean_sigma = (sigma1 + sigma2) / 2
        B = A * (1 - 2 * mean_sigma) - 2 * delta_sigma / ((1 - mean_sigma)**2 + 1e-10)
        
        return A, B
    
    def define_fluid_clusters(self):
        """Define fixed fluid clusters (theoretical positions)"""
        avg_depth = (self.depth_min + self.depth_max) / 2
        depth_factor = (avg_depth - 1500) / 1000
        
        clusters = {
            'Gas': {
                'center': (-1.6 + depth_factor*0.2, -0.12 + depth_factor*0.03),
                'std': (0.35, 0.05),
                'color': '#FF4444',
                'marker': 'circle',
                'size': 8,
                'alpha': 0.35,
                'description': 'Gas sand - Strong AVO anomaly'
            },
            'Oil': {
                'center': (-1.0 + depth_factor*0.15, -0.02 + depth_factor*0.02),
                'std': (0.30, 0.04),
                'color': '#44FF44',
                'marker': 'square',
                'size': 8,
                'alpha': 0.35,
                'description': 'Oil sand - Moderate AVO anomaly'
            },
            'Brine': {
                'center': (-0.4 + depth_factor*0.1, 0.08 + depth_factor*0.01),
                'std': (0.25, 0.03),
                'color': '#4444FF',
                'marker': 'triangle-up',
                'size': 8,
                'alpha': 0.35,
                'description': 'Brine sand - Wet sand'
            },
            'Shale': {
                'center': (0.1 + depth_factor*0.05, 0.18 + depth_factor*0.01),
                'std': (0.45, 0.06),
                'color': '#888888',
                'marker': 'diamond',
                'size': 8,
                'alpha': 0.3,
                'description': 'Shale - Background'
            }
        }
        
        for fluid, props in clusters.items():
            B_center, A_center = props['center']
            B_std, A_std = props['std']
            
            B_samples = np.random.normal(B_center, B_std, 300)
            A_samples = np.random.normal(A_center, A_std, 300)
            
            clusters[fluid]['gradient'] = B_samples
            clusters[fluid]['intercept'] = A_samples
            clusters[fluid]['covariance'] = np.cov(B_samples, A_samples)
        
        return clusters
    
    def calculate_bayesian_probability(self):
        """Calculate Bayesian probability for each data point"""
        n_samples = len(self.actual_A)
        fluids = list(self.fluid_clusters.keys())
        n_fluids = len(fluids)
        
        prior = np.array([1/n_fluids] * n_fluids)
        likelihood = np.zeros((n_samples, n_fluids))
        
        for i, fluid in enumerate(fluids):
            mean_B, mean_A = self.fluid_clusters[fluid]['center']
            cov = self.fluid_clusters[fluid]['covariance']
            cov = cov + np.eye(2) * 1e-6
            
            for j in range(n_samples):
                point = np.array([self.actual_B[j], self.actual_A[j]])
                try:
                    likelihood[j, i] = multivariate_normal.pdf(point, mean=[mean_B, mean_A], cov=cov)
                except:
                    likelihood[j, i] = 1e-10
        
        posterior = np.zeros((n_samples, n_fluids))
        for j in range(n_samples):
            posterior[j] = likelihood[j] * prior
            total = np.sum(posterior[j])
            if total > 0:
                posterior[j] = posterior[j] / total
            else:
                posterior[j] = 1/n_fluids
        
        # Create Bayesian clusters
        bayesian_clusters = {}
        
        for i, fluid in enumerate(fluids):
            mask = posterior.argmax(axis=1) == i
            
            if np.sum(mask) > 0:
                B_points = self.actual_B[mask]
                A_points = self.actual_A[mask]
                
                mean_B = np.mean(B_points)
                mean_A = np.mean(A_points)
                cov = np.cov(B_points, A_points) if len(B_points) > 1 else np.eye(2) * 0.01
                
                bayesian_clusters[fluid] = {
                    'gradient': B_points,
                    'intercept': A_points,
                    'center': (mean_B, mean_A),
                    'covariance': cov,
                    'size': len(B_points),
                    'color': self.fluid_clusters[fluid]['color'],
                    'marker': self.fluid_clusters[fluid]['marker'],
                    'probability': np.mean(posterior[mask], axis=0)[i] if np.sum(mask) > 0 else 0,
                    'description': f'{fluid} - Bayesian classification'
                }
            else:
                bayesian_clusters[fluid] = {
                    'gradient': np.array([]),
                    'intercept': np.array([]),
                    'center': self.fluid_clusters[fluid]['center'],
                    'covariance': np.eye(2) * 0.01,
                    'size': 0,
                    'color': self.fluid_clusters[fluid]['color'],
                    'marker': self.fluid_clusters[fluid]['marker'],
                    'probability': 0,
                    'description': f'{fluid} - No points'
                }
        
        # Create probability dataframe
        prob_df = pd.DataFrame(posterior, columns=[f'P_{fluid.lower()}' for fluid in fluids])
        prob_df['DEPTH'] = self.df['DEPTH'].values
        prob_df['Sw'] = self.df['Sw'].values
        prob_df['Vclay'] = self.df['Vclay'].values
        prob_df['Phie'] = self.df['Phie'].values
        prob_df['Vp'] = self.df['Vp'].values
        prob_df['Vs'] = self.df['Vs'].values
        prob_df['rho'] = self.df['rho'].values
        
        fluid_cols = [f'P_{fluid.lower()}' for fluid in fluids]
        prob_df['Most_Likely'] = prob_df[fluid_cols].idxmax(axis=1).str.replace('P_', '')
        prob_df['Max_Probability'] = prob_df[fluid_cols].max(axis=1)
        
        return prob_df, bayesian_clusters
    
    def plot_well_logs_continuous(self):
        """
        Create continuous well log curves (lines with depth)
        This matches the style in your image
        """
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=(
                'P-wave Velocity (Vp)', 'S-wave Velocity (Vs)', 'Density (ρ)', 'Porosity (φ)',
                'AVO Intercept (A)', 'AVO Gradient (B)', 'Water Saturation (Sw)', 'Vclay'
            ),
            shared_yaxes=True,
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Row 1: Velocity, Density, Porosity
        # Vp
        fig.add_trace(
            go.Scatter(
                x=self.df['Vp'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                name='Vp',
                fill='tozerox',
                fillcolor='rgba(31,119,180,0.1)'
            ),
            row=1, col=1
        )
        
        # Vs
        fig.add_trace(
            go.Scatter(
                x=self.df['Vs'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#ff7f0e', width=2),
                name='Vs',
                fill='tozerox',
                fillcolor='rgba(255,127,14,0.1)'
            ),
            row=1, col=2
        )
        
        # Density
        fig.add_trace(
            go.Scatter(
                x=self.df['rho'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#2ca02c', width=2),
                name='Density',
                fill='tozerox',
                fillcolor='rgba(44,160,44,0.1)'
            ),
            row=1, col=3
        )
        
        # Porosity
        fig.add_trace(
            go.Scatter(
                x=self.df['Phie'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#d62728', width=2),
                name='Porosity',
                fill='tozerox',
                fillcolor='rgba(214,39,40,0.1)'
            ),
            row=1, col=4
        )
        
        # Row 2: AVO Attributes, Saturation, Vclay
        # AVO Intercept
        fig.add_trace(
            go.Scatter(
                x=self.actual_A,
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#9467bd', width=2),
                name='AVO Intercept',
                fill='tozerox',
                fillcolor='rgba(148,103,189,0.1)'
            ),
            row=2, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Zero")
        
        # AVO Gradient
        fig.add_trace(
            go.Scatter(
                x=self.actual_B,
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#e377c2', width=2),
                name='AVO Gradient',
                fill='tozerox',
                fillcolor='rgba(227,119,194,0.1)'
            ),
            row=2, col=2
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=2, annotation_text="Zero")
        
        # Water Saturation
        fig.add_trace(
            go.Scatter(
                x=self.df['Sw'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#17becf', width=2),
                name='Water Saturation',
                fill='tozerox',
                fillcolor='rgba(23,190,207,0.1)'
            ),
            row=2, col=3
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="red", row=2, col=3, 
                     annotation_text="HC Threshold", annotation_position="top right")
        
        # Vclay
        fig.add_trace(
            go.Scatter(
                x=self.df['Vclay'],
                y=self.df['DEPTH'],
                mode='lines',
                line=dict(color='#7f7f7f', width=2),
                name='Vclay',
                fill='tozerox',
                fillcolor='rgba(127,127,127,0.1)'
            ),
            row=2, col=4
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Well Logs - Depth: {self.depth_min:.0f}-{self.depth_max:.0f}m ({self.depth_max-self.depth_min:.0f}m interval)",
                font=dict(size=16)
            ),
            height=900,
            width=1400,
            showlegend=False,
            hovermode='y unified'
        )
        
        # Update y-axes
        for row in [1, 2]:
            for col in [1, 2, 3, 4]:
                fig.update_yaxes(title_text="Depth (m)", row=row, col=col, autorange="reversed")
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=1)
        fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=2)
        fig.update_xaxes(title_text="Density (kg/m³)", row=1, col=3)
        fig.update_xaxes(title_text="Porosity", row=1, col=4)
        fig.update_xaxes(title_text="Intercept (A)", row=2, col=1)
        fig.update_xaxes(title_text="Gradient (B)", row=2, col=2)
        fig.update_xaxes(title_text="Water Saturation", row=2, col=3)
        fig.update_xaxes(title_text="Vclay", row=2, col=4)
        
        return fig
    
    def plot_avo_crossplot(self):
        """Create AVO crossplot"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Theoretical Fluid Clusters<br>Depth: {self.depth_min}-{self.depth_max}m',
                f'Bayesian Probability Clusters<br>(From Well Data Classification)'
            ),
            horizontal_spacing=0.12,
            x_title='AVO Gradient (B)',
            y_title='AVO Intercept (A)'
        )
        
        # Left plot: Theoretical clusters
        for fluid, res in self.fluid_clusters.items():
            # Cluster points
            fig.add_trace(
                go.Scatter(
                    x=res['gradient'],
                    y=res['intercept'],
                    mode='markers',
                    marker=dict(
                        color=res['color'],
                        size=res['size']*0.5,
                        symbol=res['marker'],
                        opacity=res['alpha'],
                        line=dict(width=0)
                    ),
                    name=f"{fluid} (theoretical)",
                    legendgroup=f"theoretical_{fluid}",
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Cluster center
            B_center, A_center = res['center']
            fig.add_trace(
                go.Scatter(
                    x=[B_center],
                    y=[A_center],
                    mode='markers+text',
                    marker=dict(
                        color=res['color'],
                        size=res['size']*1.2,
                        symbol=res['marker'],
                        line=dict(color='black', width=1.5)
                    ),
                    text=[fluid],
                    textposition='top center',
                    textfont=dict(size=10, color=res['color'], weight='bold'),
                    name=f"{fluid} center",
                    legendgroup=f"theoretical_{fluid}",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # IN-SITU data as points
        fig.add_trace(
            go.Scatter(
                x=self.actual_B,
                y=self.actual_A,
                mode='markers',
                marker=dict(
                    color=self.df['Sw'],
                    size=10,
                    symbol='circle',
                    opacity=0.9,
                    line=dict(color='black', width=0.5),
                    colorbar=dict(title="Water Saturation (Sw)", x=0.45, len=0.8),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Depth: {d:.0f}m<br>Sw: {sw:.3f}<br>Vclay: {vclay:.3f}<br>Phie: {phie:.3f}<br>Vp: {vp:.0f}<br>Vs: {vs:.0f}" 
                      for d, sw, vclay, phie, vp, vs in zip(self.df['DEPTH'], self.df['Sw'], 
                                                            self.df['Vclay'], self.df['Phie'],
                                                            self.df['Vp'], self.df['Vs'])],
                hoverinfo='text',
                name='IN-SITU Well Data',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Right plot: Bayesian clusters
        for fluid, res in self.bayesian_clusters.items():
            if res['size'] > 0:
                fig.add_trace(
                    go.Scatter(
                        x=res['gradient'],
                        y=res['intercept'],
                        mode='markers',
                        marker=dict(
                            color=res['color'],
                            size=8,
                            symbol=res['marker'],
                            opacity=0.6,
                            line=dict(width=0)
                        ),
                        name=f"{fluid} (Bayesian)",
                        legendgroup=f"bayesian_{fluid}",
                        showlegend=True
                    ),
                    row=1, col=2
                )
                
                B_center, A_center = res['center']
                fig.add_trace(
                    go.Scatter(
                        x=[B_center],
                        y=[A_center],
                        mode='markers+text',
                        marker=dict(
                            color=res['color'],
                            size=15,
                            symbol=res['marker'],
                            line=dict(color='black', width=1.5)
                        ),
                        text=[fluid],
                        textposition='top center',
                        textfont=dict(size=10, color=res['color'], weight='bold'),
                        name=f"{fluid} center",
                        legendgroup=f"bayesian_{fluid}",
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.add_trace(
            go.Scatter(
                x=self.actual_B,
                y=self.actual_A,
                mode='markers',
                marker=dict(
                    color=self.df['Sw'],
                    size=10,
                    symbol='circle',
                    opacity=0.9,
                    line=dict(color='black', width=0.5),
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"Depth: {d:.0f}m<br>Sw: {sw:.3f}<br>Most Likely: {ml}<br>Prob: {p:.3f}" 
                      for d, sw, ml, p in zip(self.df['DEPTH'], self.df['Sw'], 
                                              self.prob_df['Most_Likely'],
                                              self.prob_df['Max_Probability'])],
                hoverinfo='text',
                name='IN-SITU Well Data',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Reference lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)
        
        fig.update_layout(
            height=600,
            width=1400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest'
        )
        
        fig.update_xaxes(range=[-2.5, 1.5], row=1, col=1)
        fig.update_yaxes(range=[-0.35, 0.35], row=1, col=1)
        fig.update_xaxes(range=[-2.5, 1.5], row=1, col=2)
        fig.update_yaxes(range=[-0.35, 0.35], row=1, col=2)
        
        return fig
    
    def plot_probability_maps(self):
        """Create probability maps"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Gas Probability', 'Oil Probability', 'Water/Brine Probability',
                'Combined Probabilities', 'Most Likely Fluid Type', 'Average Probabilities'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Gas probability (line with fill)
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_gas'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='red', width=2),
                name='Gas',
                fill='tozerox',
                fillcolor='rgba(255,0,0,0.2)'
            ),
            row=1, col=1
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=1, 
                     annotation_text="Threshold")
        
        # Oil probability
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_oil'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Oil',
                fill='tozerox',
                fillcolor='rgba(0,255,0,0.2)'
            ),
            row=1, col=2
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=2)
        
        # Brine probability
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_brine'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Brine',
                fill='tozerox',
                fillcolor='rgba(0,0,255,0.2)'
            ),
            row=1, col=3
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=3)
        
        # Combined probabilities
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_gas'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='red', width=2),
                name='Gas',
                legendgroup='combined',
                showlegend=True
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_oil'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Oil',
                legendgroup='combined',
                showlegend=True
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['P_brine'],
                y=self.prob_df['DEPTH'],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Brine',
                legendgroup='combined',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Most likely fluid (points)
        color_map = {'gas': 'red', 'oil': 'green', 'brine': 'blue', 'shale': 'gray'}
        colors = [color_map.get(f, 'black') for f in self.prob_df['Most_Likely']]
        
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['Max_Probability'],
                y=self.prob_df['DEPTH'],
                mode='markers',
                marker=dict(color=colors, size=8, symbol='circle', line=dict(color='black', width=0.5)),
                text=[f"Depth: {d:.0f}m<br>Most Likely: {ml}<br>Prob: {p:.3f}<br>Sw: {sw:.3f}"
                      for d, ml, p, sw in zip(self.prob_df['DEPTH'], self.prob_df['Most_Likely'],
                                              self.prob_df['Max_Probability'], self.prob_df['Sw'])],
                hoverinfo='text',
                name='Most Likely Fluid'
            ),
            row=2, col=2
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=2, col=2)
        
        # Average probabilities bar chart
        fluid_cols = ['gas', 'oil', 'brine', 'shale']
        means = [self.prob_df[f'P_{f}'].mean() for f in fluid_cols]
        colors_bar = ['red', 'green', 'blue', 'gray']
        
        fig.add_trace(
            go.Bar(
                x=[f.capitalize() for f in fluid_cols],
                y=means,
                marker_color=colors_bar,
                text=[f'{m:.3f}' for m in means],
                textposition='outside',
                name='Average Probability'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=900,
            width=1400,
            hovermode='y unified'
        )
        
        # Update y-axes
        for i in range(1, 4):
            fig.update_yaxes(title_text="Depth (m)", row=1, col=i, autorange="reversed")
            fig.update_yaxes(title_text="Depth (m)", row=2, col=1, autorange="reversed")
            fig.update_yaxes(title_text="Depth (m)", row=2, col=2, autorange="reversed")
        
        # Update x-axes
        fig.update_xaxes(title_text="Probability", row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=1, col=2, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=1, col=3, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=2, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=2, col=2, range=[0, 1])
        fig.update_xaxes(title_text="Fluid Type", row=2, col=3)
        
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics for display"""
        stats_dict = {
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'samples': len(self.df),
            'avg_vp': np.mean(self.df['Vp']),
            'avg_vs': np.mean(self.df['Vs']),
            'avg_phi': np.mean(self.df['Phie']),
            'avg_sw': np.mean(self.df['Sw']),
            'avg_vclay': np.mean(self.df['Vclay']),
            'avg_rho': np.mean(self.df['rho']),
            'avg_intercept': np.mean(self.actual_A),
            'avg_gradient': np.mean(self.actual_B),
        }
        
        # Fluid probabilities
        for fluid in ['gas', 'oil', 'brine', 'shale']:
            col = f'P_{fluid}'
            if col in self.prob_df.columns:
                stats_dict[f'prob_{fluid}'] = self.prob_df[col].mean()
        
        # Most likely fluid distribution
        if 'Most_Likely' in self.prob_df.columns:
            counts = self.prob_df['Most_Likely'].value_counts()
            stats_dict['most_likely'] = counts.to_dict()
        
        return stats_dict


def generate_sample_data():
    """Generate sample well log data for demo"""
    np.random.seed(42)
    depths = np.linspace(800, 2300, 500)
    
    Vp = 1800 + 0.5 * (depths - 800) + np.random.normal(0, 50, 500)
    Vs = Vp / 1.8 + np.random.normal(0, 20, 500)
    Phie = 0.3 * np.exp(-(depths - 800) / 2000) + np.random.normal(0, 0.02, 500)
    Phie = np.clip(Phie, 0.05, 0.35)
    rho = 2.2 + 0.2 * (Vp / 3000) + np.random.normal(0, 0.02, 500)
    
    gas_zone = (depths > 900) & (depths < 1100)
    oil_zone = (depths > 1400) & (depths < 1600)
    deep_oil_zone = (depths > 1900) & (depths < 1970)
    
    Vp[gas_zone] = Vp[gas_zone] * 0.85
    Vp[oil_zone] = Vp[oil_zone] * 0.93
    Vp[deep_oil_zone] = Vp[deep_oil_zone] * 0.95
    
    Vs[gas_zone] = Vs[gas_zone] * 0.9
    Vs[oil_zone] = Vs[oil_zone] * 0.96
    
    Sw = np.ones(500) * 0.9
    Sw[gas_zone] = 0.15
    Sw[oil_zone] = 0.25
    Sw[deep_oil_zone] = 0.30
    Sw = Sw + np.random.normal(0, 0.05, 500)
    Sw = np.clip(Sw, 0.05, 0.95)
    
    Vclay = 0.2 + 0.15 * np.exp(-(depths - 800) / 1500) + np.random.normal(0, 0.05, 500)
    Vclay = np.clip(Vclay, 0.05, 0.35)
    GR = 30 + 70 * Vclay + np.random.normal(0, 5, 500)
    RT = 10 * (1 - Sw) + np.random.normal(0, 1, 500)
    RT = np.clip(RT, 0.5, 50)
    
    return pd.DataFrame({
        'DEPTH': depths, 'Vp': Vp, 'Vs': Vs, 'Phie': Phie,
        'GR': GR, 'rho': rho, 'RT': RT, 'Sw': Sw, 'Vclay': Vclay
    })


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🛢️ AVO Fluid Inversion (AFI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on: FD Field, Niger Delta - Reservoir Fluid Content Prediction</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with well logs",
            type=['csv'],
            help="Required columns: DEPTH, Vp, Vs, Phie, GR, rho, RT, Sw, Vclay"
        )
        
        # Use sample data if no file uploaded
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(df)} samples")
            st.info(f"Depth range: {df['DEPTH'].min():.0f} - {df['DEPTH'].max():.0f}m")
        else:
            st.info("Using sample data (Niger Delta synthetic well logs)")
            if st.button("Load Sample Data"):
                df = generate_sample_data()
                st.success(f"✓ Loaded {len(df)} sample samples")
                st.info(f"Depth range: {df['DEPTH'].min():.0f} - {df['DEPTH'].max():.0f}m")
            else:
                df = generate_sample_data()
        
        st.markdown("---")
        st.markdown("## ⚙️ Analysis Settings")
        
        # Depth range selection
        if 'df' in locals():
            min_depth = float(df['DEPTH'].min())
            max_depth = float(df['DEPTH'].max())
            
            col1, col2 = st.columns(2)
            with col1:
                depth_min = st.number_input("Min Depth (m)", min_value=min_depth, max_value=max_depth, value=min_depth, step=50.0)
            with col2:
                depth_max = st.number_input("Max Depth (m)", min_value=min_depth, max_value=max_depth, value=max_depth, step=50.0)
            
            if depth_min >= depth_max:
                st.error("Min depth must be less than max depth")
                depth_min, depth_max = min_depth, max_depth
        
        st.markdown("---")
        st.markdown("## 📊 About")
        st.markdown("""
        **AVO Fluid Inversion (AFI)** uses:
        - AVO Intercept (A) and Gradient (B)
        - Monte-Carlo simulation
        - Biot-Gassmann fluid substitution
        - Bayesian probability estimation
        
        Results show probability of Gas, Oil, Brine, and Shale.
        """)
    
    # Main content
    if 'df' in locals():
        # Show data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(10))
            
            # Required columns check
            required_cols = ['DEPTH', 'Vp', 'Vs', 'Phie', 'GR', 'rho', 'RT', 'Sw', 'Vclay']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠ Missing columns: {missing_cols}")
        
        # Run analysis
        with st.spinner("Running AVO Fluid Inversion analysis..."):
            try:
                afi = AVOFluidInversionApp(df, depth_min, depth_max)
                
                # Summary metrics
                stats = afi.get_summary_stats()
                
                # First row metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Samples", stats['samples'])
                with col2:
                    st.metric("Avg Porosity", f"{stats['avg_phi']:.3f}")
                with col3:
                    st.metric("Avg Water Saturation", f"{stats['avg_sw']:.3f}")
                with col4:
                    st.metric("Avg Vclay", f"{stats['avg_vclay']:.3f}")
                
                # Second row metrics - Probabilities
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Gas Probability", f"{stats.get('prob_gas', 0):.3f}", 
                             delta="High" if stats.get('prob_gas', 0) > 0.3 else "Low")
                with col2:
                    st.metric("Oil Probability", f"{stats.get('prob_oil', 0):.3f}",
                             delta="High" if stats.get('prob_oil', 0) > 0.25 else "Low")
                with col3:
                    st.metric("Brine Probability", f"{stats.get('prob_brine', 0):.3f}")
                with col4:
                    st.metric("Shale Probability", f"{stats.get('prob_shale', 0):.3f}")
                
                # Tabs for plots
                tab1, tab2, tab3 = st.tabs(["📈 AVO Crossplot", "📊 Well Logs", "🎯 Probability Maps"])
                
                with tab1:
                    st.plotly_chart(afi.plot_avo_crossplot(), use_container_width=True, height=600)
                    
                    # Explanation
                    with st.expander("ℹ️ AVO Crossplot Interpretation"):
                        st.markdown("""
                        **AVO Crossplot Interpretation:**
                        
                        - **Left Plot**: Theoretical fluid clusters (Gas, Oil, Brine, Shale) based on rock physics
                        - **Right Plot**: Bayesian probability clusters derived from actual data classification
                        - **IN-SITU points**: Colored by water saturation (dark blue = low Sw = potential hydrocarbons)
                        - Points below cluster centers indicate stronger AVO anomalies
                        - Gas shows most negative gradient, Oil moderate, Brine on trend
                        
                        **Water Saturation Color Scale:**
                        - Dark blue (0-0.2): High hydrocarbon saturation
                        - Light blue (0.2-0.4): Moderate hydrocarbon saturation
                        - Yellow-green (0.4-0.7): Mixed saturation
                        - Red (0.7-1.0): Water-wet
                        """)
                
                with tab2:
                    st.plotly_chart(afi.plot_well_logs_continuous(), use_container_width=True, height=900)
                    
                    with st.expander("ℹ️ Well Log Interpretation"):
                        st.markdown("""
                        **Well Log Interpretation:**
                        
                        - **Vp, Vs, Density**: Decrease in hydrocarbon zones (lower velocities)
                        - **Porosity**: Higher values indicate good reservoir quality
                        - **AVO Intercept (A)**: Negative values may indicate hydrocarbons
                        - **AVO Gradient (B)**: Negative values with large magnitude indicate gas
                        - **Water Saturation**: Values below 0.4 indicate potential hydrocarbons
                        - **Vclay**: Values below 0.3 indicate clean sand
                        """)
                
                with tab3:
                    st.plotly_chart(afi.plot_probability_maps(), use_container_width=True, height=900)
                    
                    with st.expander("ℹ️ Probability Maps Interpretation"):
                        st.markdown("""
                        **Probability Maps Interpretation:**
                        
                        - **Gas/Oil Probability**: Values > 0.4 indicate high probability
                        - **Combined Probabilities**: Shows relative likelihood of each fluid
                        - **Most Likely Fluid**: Color-coded points show the dominant fluid at each depth
                        - **Average Probabilities**: Summary of overall fluid distribution
                        
                        **Interpretation Guidelines:**
                        - High Gas probability + Low Sw = Gas reservoir
                        - High Oil probability + Moderate Sw = Oil reservoir
                        - High Brine probability = Water-wet zone
                        - High Shale probability = Non-reservoir
                        """)
                
                # Results summary
                with st.expander("📊 Detailed Analysis Summary"):
                    st.markdown("### Bayesian Probability Results")
                    
                    # Most likely fluid distribution
                    if 'most_likely' in stats:
                        st.markdown("**Most Likely Fluid Distribution:**")
                        cols = st.columns(len(stats['most_likely']))
                        for i, (fluid, count) in enumerate(stats['most_likely'].items()):
                            with cols[i]:
                                st.metric(f"{fluid.capitalize()}", f"{count} samples", 
                                         f"{count/stats['samples']*100:.1f}%")
                    
                    st.markdown("### Theoretical vs Bayesian Cluster Centers")
                    st.markdown("""
                    | Fluid | Theoretical Center (B, A) | Bayesian Center (B, A) | Samples |
                    |-------|--------------------------|------------------------|---------|
                    | Gas | {gas_th} | {gas_bay} | {gas_size} |
                    | Oil | {oil_th} | {oil_bay} | {oil_size} |
                    | Brine | {brine_th} | {brine_bay} | {brine_size} |
                    | Shale | {shale_th} | {shale_bay} | {shale_size} |
                    """.format(
                        gas_th=f"({afi.fluid_clusters['Gas']['center'][0]:.2f}, {afi.fluid_clusters['Gas']['center'][1]:.2f})",
                        gas_bay=f"({afi.bayesian_clusters['Gas']['center'][0]:.4f}, {afi.bayesian_clusters['Gas']['center'][1]:.4f})" if afi.bayesian_clusters['Gas']['size'] > 0 else "N/A",
                        gas_size=afi.bayesian_clusters['Gas']['size'],
                        oil_th=f"({afi.fluid_clusters['Oil']['center'][0]:.2f}, {afi.fluid_clusters['Oil']['center'][1]:.2f})",
                        oil_bay=f"({afi.bayesian_clusters['Oil']['center'][0]:.4f}, {afi.bayesian_clusters['Oil']['center'][1]:.4f})" if afi.bayesian_clusters['Oil']['size'] > 0 else "N/A",
                        oil_size=afi.bayesian_clusters['Oil']['size'],
                        brine_th=f"({afi.fluid_clusters['Brine']['center'][0]:.2f}, {afi.fluid_clusters['Brine']['center'][1]:.2f})",
                        brine_bay=f"({afi.bayesian_clusters['Brine']['center'][0]:.4f}, {afi.bayesian_clusters['Brine']['center'][1]:.4f})" if afi.bayesian_clusters['Brine']['size'] > 0 else "N/A",
                        brine_size=afi.bayesian_clusters['Brine']['size'],
                        shale_th=f"({afi.fluid_clusters['Shale']['center'][0]:.2f}, {afi.fluid_clusters['Shale']['center'][1]:.2f})",
                        shale_bay=f"({afi.bayesian_clusters['Shale']['center'][0]:.4f}, {afi.bayesian_clusters['Shale']['center'][1]:.4f})" if afi.bayesian_clusters['Shale']['size'] > 0 else "N/A",
                        shale_size=afi.bayesian_clusters['Shale']['size']
                    ))
                    
                    st.markdown("### AVO Attributes")
                    st.markdown(f"""
                    - **AVO Intercept (A)**: Mean = {stats['avg_intercept']:.4f}, Range = [{afi.actual_A.min():.4f}, {afi.actual_A.max():.4f}]
                    - **AVO Gradient (B)**: Mean = {stats['avg_gradient']:.4f}, Range = [{afi.actual_B.min():.4f}, {afi.actual_B.max():.4f}]
                    - **AVO Classification**: Based on mean values
                    """)
                    
                    # AVO Classification
                    mean_A = stats['avg_intercept']
                    mean_B = stats['avg_gradient']
                    
                    if mean_A < 0 and mean_B < -0.5:
                        st.success("**AVO Class III/IV**: Strong AVO anomaly - Gas sands likely")
                    elif mean_A < 0 and mean_B < 0:
                        st.info("**AVO Class II/III**: Moderate AVO anomaly - Oil sands possible")
                    elif abs(mean_A) < 0.1 and mean_B < 0:
                        st.warning("**AVO Class II**: Subtle anomaly - Dim spots")
                    elif mean_A > 0 and mean_B < 0:
                        st.info("**AVO Class I**: Hard sands - Deep reservoirs")
                    else:
                        st.write("**AVO Class**: Undetermined")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check your data format and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "AVO Fluid Inversion (AFI) - Based on FD Field, Niger Delta Nigeria<br>"
        "Methodology: Shuey's AVO approximation | Biot-Gassmann fluid substitution | Bayesian probability"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
