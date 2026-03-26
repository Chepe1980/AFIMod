"""
AVO Fluid Inversion (AFI) - STREAMLIT WEB APPLICATION
Based on: "AVO FLUID INVERSION (AFI) TECHNIQUE AS A TOOL TO PREDICT RESERVOIR 
FLUID CONTENT USING DATA FROM FD FIELD, ONSHORE NIGER DELTA NIGERIA"

Features:
- Upload CSV well log data
- Interactive depth range selection
- Fluid clusters based on actual log data (Vp, Vs, density, porosity, Sw)
- Theoretical fluid substitution using Biot-Gassmann
- Bayesian probability classification
- Interactive Plotly visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
from scipy import stats
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
    """AVO Fluid Inversion for Streamlit App with Data-Driven Clusters"""
    
    def __init__(self, df, depth_min, depth_max):
        """Initialize with well log data"""
        mask = (df['DEPTH'] >= depth_min) & (df['DEPTH'] <= depth_max)
        self.df = df[mask].copy().sort_values('DEPTH')
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.df_full = df.copy()
        
        # Rock matrix properties (sandstone)
        self.K_matrix = 37e9  # Bulk modulus of quartz [Pa]
        self.rho_matrix = 2650  # Matrix density [kg/m^3]
        self.mu_matrix = 44e9  # Shear modulus [Pa]
        
        # Fluid properties
        self.fluid_props = {
            'brine': {'K': 2.5e9, 'rho': 1040, 'color': '#4444FF', 'marker': 'triangle-up'},
            'oil': {'K': 1.2e9, 'rho': 800, 'color': '#44FF44', 'marker': 'square'},
            'gas': {'K': 0.1e9, 'rho': 200, 'color': '#FF4444', 'marker': 'circle'}
        }
        
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
        
        # Generate fluid clusters based on log data
        self.fluid_clusters = self.generate_data_driven_clusters()
        
        # Calculate Bayesian probability
        self.prob_df, self.bayesian_clusters = self.calculate_bayesian_probability()
        
        # Create rock physics templates
        self.rock_physics_template = self.create_rock_physics_template()
    
    def calculate_avo_attributes(self):
        """Calculate AVO Intercept (A) and Gradient (B) using Shuey's approximation"""
        # Acoustic impedance
        I1 = self.rho_back * self.Vp_back
        I2 = self.df['rho'].values * self.df['Vp'].values
        
        # Intercept A - reflection coefficient at zero offset
        A = (I2 - I1) / (I2 + I1 + 1e-10)
        
        # Poisson's ratio for background
        VpVs_back = self.Vp_back / self.Vs_back
        sigma1 = (0.5 * VpVs_back**2 - 1) / (VpVs_back**2 - 1)
        
        # Poisson's ratio for each sample
        VpVs = self.df['Vp'].values / (self.df['Vs'].values + 1e-10)
        sigma2 = np.zeros_like(VpVs)
        mask = VpVs > 0
        sigma2[mask] = (0.5 * VpVs[mask]**2 - 1) / (VpVs[mask]**2 - 1)
        sigma2[~mask] = sigma1
        
        # Gradient B using Shuey's approximation
        delta_sigma = sigma2 - sigma1
        mean_sigma = (sigma1 + sigma2) / 2
        B = A * (1 - 2 * mean_sigma) - 2 * delta_sigma / ((1 - mean_sigma)**2 + 1e-10)
        
        return A, B
    
    def gassmann_fluid_substitution(self, phi, K_dry, mu_dry, K_fluid, rho_fluid):
        """Perform Biot-Gassmann fluid substitution"""
        # Saturated bulk modulus
        K_sat = K_dry + (1 - K_dry/self.K_matrix)**2 / (phi/K_fluid + (1-phi)/self.K_matrix - K_dry/self.K_matrix**2 + 1e-10)
        
        # Saturated density
        rho_sat = (1 - phi) * self.rho_matrix + phi * rho_fluid
        
        # Saturated velocities
        Vp_sat = np.sqrt((K_sat + 4/3 * mu_dry) / rho_sat)
        Vs_sat = np.sqrt(mu_dry / rho_sat)
        
        return Vp_sat, Vs_sat, rho_sat
    
    def estimate_dry_rock_moduli(self):
        """Estimate dry rock moduli from well log data using critical porosity model"""
        phi = self.df['Phie'].values
        Vp = self.df['Vp'].values
        Vs = self.df['Vs'].values
        rho = self.df['rho'].values
        
        # Calculate saturated moduli
        mu_sat = rho * Vs**2
        K_sat = rho * Vp**2 - 4/3 * mu_sat
        
        # Critical porosity model (Nur, 1995)
        phi_c = 0.4  # Critical porosity for sandstone
        
        # Dry rock moduli
        K_dry = K_sat * (1 - phi/phi_c) / (1 - phi/phi_c * K_sat/self.K_matrix + 1e-10)
        mu_dry = mu_sat * (1 - phi/phi_c) / (1 - phi/phi_c * mu_sat/self.mu_matrix + 1e-10)
        
        # Ensure physical values
        K_dry = np.clip(K_dry, 1e9, self.K_matrix)
        mu_dry = np.clip(mu_dry, 1e8, self.mu_matrix)
        
        return K_dry, mu_dry, phi
    
    def generate_data_driven_clusters(self):
        """
        Generate fluid clusters based on actual well log data
        Using rock physics and fluid substitution
        """
        print("\n" + "="*60)
        print("Generating Data-Driven Fluid Clusters")
        print("="*60)
        
        # Estimate dry rock moduli from actual data
        K_dry, mu_dry, phi = self.estimate_dry_rock_moduli()
        
        # Use average properties for cluster generation
        avg_phi = np.mean(phi)
        avg_K_dry = np.mean(K_dry)
        avg_mu_dry = np.mean(mu_dry)
        
        clusters = {}
        
        # Generate clusters for each fluid type
        for fluid_name, props in self.fluid_props.items():
            print(f"  Generating {fluid_name.upper()} cluster...")
            
            intercepts = []
            gradients = []
            
            # Generate multiple realizations with porosity variation
            for i in range(300):
                # Perturb porosity around average
                phi_pert = avg_phi * (1 + np.random.normal(0, 0.1))
                phi_pert = np.clip(phi_pert, 0.05, 0.35)
                
                # Perturb dry rock moduli
                K_dry_pert = avg_K_dry * (1 + np.random.normal(0, 0.05))
                mu_dry_pert = avg_mu_dry * (1 + np.random.normal(0, 0.05))
                
                # Gassmann fluid substitution
                Vp_sat, Vs_sat, rho_sat = self.gassmann_fluid_substitution(
                    phi_pert, K_dry_pert, mu_dry_pert, 
                    props['K'], props['rho']
                )
                
                # Calculate AVO attributes
                I1 = self.rho_back * self.Vp_back
                I2 = rho_sat * Vp_sat
                A = (I2 - I1) / (I2 + I1 + 1e-10)
                
                # Poisson's ratio
                VpVs = Vp_sat / Vs_sat
                sigma = (0.5 * VpVs**2 - 1) / (VpVs**2 - 1)
                sigma_back = (0.5 * (self.Vp_back/self.Vs_back)**2 - 1) / ((self.Vp_back/self.Vs_back)**2 - 1)
                
                # Gradient
                delta_sigma = sigma - sigma_back
                mean_sigma = (sigma + sigma_back) / 2
                B = A * (1 - 2 * mean_sigma) - 2 * delta_sigma / ((1 - mean_sigma)**2 + 1e-10)
                
                intercepts.append(A)
                gradients.append(B)
            
            # Convert to arrays
            intercepts = np.array(intercepts)
            gradients = np.array(gradients)
            
            # Remove outliers
            mask = (np.abs(intercepts) < 0.5) & (np.abs(gradients) < 3)
            intercepts = intercepts[mask]
            gradients = gradients[mask]
            
            clusters[fluid_name.capitalize()] = {
                'gradient': gradients,
                'intercept': intercepts,
                'center': (np.mean(gradients), np.mean(intercepts)),
                'std': (np.std(gradients), np.std(intercepts)),
                'covariance': np.cov(gradients, intercepts),
                'color': props['color'],
                'marker': props['marker'],
                'size': len(intercepts),
                'alpha': 0.35,
                'description': f'{fluid_name.capitalize()} sand - Fluid substitution from log data'
            }
            
            print(f"    Center: ({np.mean(gradients):.3f}, {np.mean(intercepts):.3f})")
            print(f"    Size: {len(intercepts)} samples")
        
        # Add Shale cluster (based on high Vclay points)
        shale_mask = self.df['Vclay'] > 0.5
        if np.sum(shale_mask) > 10:
            shale_A = self.actual_A[shale_mask]
            shale_B = self.actual_B[shale_mask]
            
            clusters['Shale'] = {
                'gradient': shale_B,
                'intercept': shale_A,
                'center': (np.mean(shale_B), np.mean(shale_A)),
                'std': (np.std(shale_B), np.std(shale_A)),
                'covariance': np.cov(shale_B, shale_A),
                'color': '#888888',
                'marker': 'diamond',
                'size': len(shale_A),
                'alpha': 0.3,
                'description': 'Shale - High clay content'
            }
            print(f"  Shale: center=({np.mean(shale_B):.3f}, {np.mean(shale_A):.3f}), size={len(shale_A)}")
        else:
            # Default shale cluster
            clusters['Shale'] = {
                'gradient': np.random.normal(-0.2, 0.3, 300),
                'intercept': np.random.normal(0.1, 0.04, 300),
                'center': (-0.2, 0.1),
                'std': (0.3, 0.04),
                'covariance': np.array([[0.09, 0], [0, 0.0016]]),
                'color': '#888888',
                'marker': 'diamond',
                'size': 300,
                'alpha': 0.3,
                'description': 'Shale - Background'
            }
        
        return clusters
    
    def create_rock_physics_template(self):
        """Create rock physics template based on actual data"""
        # Get average porosity from actual data
        avg_phi = np.mean(self.df['Phie'])
        
        # Estimate dry rock moduli
        K_dry, mu_dry, _ = self.estimate_dry_rock_moduli()
        avg_K_dry = np.mean(K_dry)
        avg_mu_dry = np.mean(mu_dry)
        
        template = {}
        
        # Generate template for each fluid
        for fluid_name, props in self.fluid_props.items():
            # Vp/Vs ratio vs Vp
            phi_range = np.linspace(0.05, 0.35, 50)
            Vp_values = []
            Vs_values = []
            
            for phi in phi_range:
                Vp_sat, Vs_sat, _ = self.gassmann_fluid_substitution(
                    phi, avg_K_dry, avg_mu_dry, props['K'], props['rho']
                )
                Vp_values.append(Vp_sat)
                Vs_values.append(Vs_sat)
            
            template[fluid_name] = {
                'phi': phi_range,
                'Vp': np.array(Vp_values),
                'Vs': np.array(Vs_values),
                'VpVs': np.array(Vp_values) / np.array(Vs_values),
                'color': props['color']
            }
        
        return template
    
    def calculate_bayesian_probability(self):
        """Calculate Bayesian probability for each data point based on fluid clusters"""
        print("\n" + "="*60)
        print("Bayesian Probability Calculation")
        print("="*60)
        
        n_samples = len(self.actual_A)
        fluids = list(self.fluid_clusters.keys())
        n_fluids = len(fluids)
        
        # Prior probabilities (based on cluster sizes)
        total_points = sum(self.fluid_clusters[f]['size'] for f in fluids)
        prior = np.array([self.fluid_clusters[f]['size'] / total_points for f in fluids])
        
        # Calculate likelihoods
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
        
        # Posterior probability (Bayes' theorem)
        posterior = np.zeros((n_samples, n_fluids))
        for j in range(n_samples):
            posterior[j] = likelihood[j] * prior
            total = np.sum(posterior[j])
            if total > 0:
                posterior[j] = posterior[j] / total
            else:
                posterior[j] = 1/n_fluids
        
        # Create Bayesian clusters (reclassified data)
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
                    'probability': np.mean(posterior[mask], axis=0)[i],
                    'description': f'{fluid} - Bayesian classification'
                }
                print(f"  {fluid}: center=({mean_B:.4f}, {mean_A:.4f}), size={len(B_points)}")
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
                print(f"  {fluid}: No points assigned")
        
        # Create probability dataframe
        prob_df = pd.DataFrame(posterior, columns=[f'P_{fluid.lower()}' for fluid in fluids])
        prob_df['DEPTH'] = self.df['DEPTH'].values
        prob_df['Sw'] = self.df['Sw'].values
        prob_df['Vclay'] = self.df['Vclay'].values
        prob_df['Phie'] = self.df['Phie'].values
        prob_df['Vp'] = self.df['Vp'].values
        prob_df['Vs'] = self.df['Vs'].values
        
        fluid_cols = [f'P_{fluid.lower()}' for fluid in fluids]
        prob_df['Most_Likely'] = prob_df[fluid_cols].idxmax(axis=1).str.replace('P_', '')
        prob_df['Max_Probability'] = prob_df[fluid_cols].max(axis=1)
        
        return prob_df, bayesian_clusters
    
    def plot_well_logs_continuous(self):
        """Create continuous well log curves"""
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
        
        # Row 1
        fig.add_trace(go.Scatter(x=self.df['Vp'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#1f77b4', width=2), name='Vp',
                                 fill='tozerox', fillcolor='rgba(31,119,180,0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=self.df['Vs'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#ff7f0e', width=2), name='Vs',
                                 fill='tozerox', fillcolor='rgba(255,127,14,0.1)'), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=self.df['rho'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#2ca02c', width=2), name='Density',
                                 fill='tozerox', fillcolor='rgba(44,160,44,0.1)'), row=1, col=3)
        
        fig.add_trace(go.Scatter(x=self.df['Phie'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#d62728', width=2), name='Porosity',
                                 fill='tozerox', fillcolor='rgba(214,39,40,0.1)'), row=1, col=4)
        
        # Row 2
        fig.add_trace(go.Scatter(x=self.actual_A, y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#9467bd', width=2), name='AVO Intercept',
                                 fill='tozerox', fillcolor='rgba(148,103,189,0.1)'), row=2, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=self.actual_B, y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#e377c2', width=2), name='AVO Gradient',
                                 fill='tozerox', fillcolor='rgba(227,119,194,0.1)'), row=2, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.add_trace(go.Scatter(x=self.df['Sw'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#17becf', width=2), name='Water Saturation',
                                 fill='tozerox', fillcolor='rgba(23,190,207,0.1)'), row=2, col=3)
        fig.add_vline(x=0.4, line_dash="dash", line_color="red", row=2, col=3)
        
        fig.add_trace(go.Scatter(x=self.df['Vclay'], y=self.df['DEPTH'], mode='lines',
                                 line=dict(color='#7f7f7f', width=2), name='Vclay',
                                 fill='tozerox', fillcolor='rgba(127,127,127,0.1)'), row=2, col=4)
        
        fig.update_layout(title=f"Well Logs - Depth: {self.depth_min:.0f}-{self.depth_max:.0f}m", 
                         height=900, width=1400, showlegend=False, hovermode='y unified')
        
        for row in [1, 2]:
            for col in [1, 2, 3, 4]:
                fig.update_yaxes(title_text="Depth (m)", row=row, col=col, autorange="reversed")
        
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
        """Create AVO crossplot with data-driven clusters"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Data-Driven Fluid Clusters<br>(From Rock Physics & Fluid Substitution)',
                f'Bayesian Probability Clusters<br>(From Well Data Classification)'
            ),
            horizontal_spacing=0.12,
            x_title='AVO Gradient (B)',
            y_title='AVO Intercept (A)'
        )
        
        # Left plot: Data-driven clusters
        for fluid, res in self.fluid_clusters.items():
            # Cluster points
            fig.add_trace(
                go.Scatter(
                    x=res['gradient'], y=res['intercept'],
                    mode='markers',
                    marker=dict(color=res['color'], size=res['size']*0.5,
                               symbol=res['marker'], opacity=res['alpha'],
                               line=dict(width=0)),
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
                    x=[B_center], y=[A_center],
                    mode='markers+text',
                    marker=dict(color=res['color'], size=15,
                               symbol=res['marker'], line=dict(color='black', width=1.5)),
                    text=[fluid], textposition='top center',
                    textfont=dict(size=10, color=res['color'], weight='bold'),
                    name=f"{fluid} center",
                    legendgroup=f"theoretical_{fluid}",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # IN-SITU data
        fig.add_trace(
            go.Scatter(
                x=self.actual_B, y=self.actual_A,
                mode='markers',
                marker=dict(color=self.df['Sw'], size=10, symbol='circle',
                           opacity=0.9, line=dict(color='black', width=0.5),
                           colorbar=dict(title="Water Saturation (Sw)", x=0.45, len=0.8),
                           colorscale='Viridis', showscale=True),
                text=[f"Depth: {d:.0f}m<br>Sw: {sw:.3f}<br>Vclay: {vclay:.3f}<br>Phie: {phie:.3f}"
                      for d, sw, vclay, phie in zip(self.df['DEPTH'], self.df['Sw'],
                                                    self.df['Vclay'], self.df['Phie'])],
                hoverinfo='text', name='IN-SITU Well Data', showlegend=True
            ),
            row=1, col=1
        )
        
        # Right plot: Bayesian clusters
        for fluid, res in self.bayesian_clusters.items():
            if res['size'] > 0:
                fig.add_trace(
                    go.Scatter(
                        x=res['gradient'], y=res['intercept'],
                        mode='markers',
                        marker=dict(color=res['color'], size=8, symbol=res['marker'],
                                   opacity=0.6, line=dict(width=0)),
                        name=f"{fluid} (Bayesian)",
                        legendgroup=f"bayesian_{fluid}",
                        showlegend=True
                    ),
                    row=1, col=2
                )
                
                B_center, A_center = res['center']
                fig.add_trace(
                    go.Scatter(
                        x=[B_center], y=[A_center],
                        mode='markers+text',
                        marker=dict(color=res['color'], size=15, symbol=res['marker'],
                                   line=dict(color='black', width=1.5)),
                        text=[fluid], textposition='top center',
                        textfont=dict(size=10, color=res['color'], weight='bold'),
                        name=f"{fluid} center",
                        legendgroup=f"bayesian_{fluid}",
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.add_trace(
            go.Scatter(
                x=self.actual_B, y=self.actual_A,
                mode='markers',
                marker=dict(color=self.df['Sw'], size=10, symbol='circle',
                           opacity=0.9, line=dict(color='black', width=0.5),
                           colorscale='Viridis', showscale=False),
                text=[f"Depth: {d:.0f}m<br>Sw: {sw:.3f}<br>Most Likely: {ml}<br>Prob: {p:.3f}"
                      for d, sw, ml, p in zip(self.df['DEPTH'], self.df['Sw'],
                                              self.prob_df['Most_Likely'],
                                              self.prob_df['Max_Probability'])],
                hoverinfo='text', name='IN-SITU Well Data', showlegend=False
            ),
            row=1, col=2
        )
        
        # Reference lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)
        
        fig.update_layout(height=600, width=1400, showlegend=True,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                         hovermode='closest')
        
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
            vertical_spacing=0.12, horizontal_spacing=0.1
        )
        
        # Gas probability
        fig.add_trace(go.Scatter(x=self.prob_df['P_gas'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='red', width=2),
                                 fill='tozerox', fillcolor='rgba(255,0,0,0.2)',
                                 name='Gas'), row=1, col=1)
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=1)
        
        # Oil probability
        fig.add_trace(go.Scatter(x=self.prob_df['P_oil'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='green', width=2),
                                 fill='tozerox', fillcolor='rgba(0,255,0,0.2)',
                                 name='Oil'), row=1, col=2)
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=2)
        
        # Brine probability
        fig.add_trace(go.Scatter(x=self.prob_df['P_brine'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='blue', width=2),
                                 fill='tozerox', fillcolor='rgba(0,0,255,0.2)',
                                 name='Brine'), row=1, col=3)
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=1, col=3)
        
        # Combined probabilities
        fig.add_trace(go.Scatter(x=self.prob_df['P_gas'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='red', width=2),
                                 name='Gas', legendgroup='combined', showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.prob_df['P_oil'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='green', width=2),
                                 name='Oil', legendgroup='combined', showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.prob_df['P_brine'], y=self.prob_df['DEPTH'],
                                 mode='lines', line=dict(color='blue', width=2),
                                 name='Brine', legendgroup='combined', showlegend=True), row=2, col=1)
        
        # Most likely fluid
        color_map = {'gas': 'red', 'oil': 'green', 'brine': 'blue', 'shale': 'gray'}
        colors = [color_map.get(f, 'black') for f in self.prob_df['Most_Likely']]
        
        fig.add_trace(
            go.Scatter(
                x=self.prob_df['Max_Probability'], y=self.prob_df['DEPTH'],
                mode='markers',
                marker=dict(color=colors, size=8, symbol='circle', line=dict(color='black', width=0.5)),
                text=[f"Depth: {d:.0f}m<br>Most Likely: {ml}<br>Prob: {p:.3f}<br>Sw: {sw:.3f}"
                      for d, ml, p, sw in zip(self.prob_df['DEPTH'], self.prob_df['Most_Likely'],
                                              self.prob_df['Max_Probability'], self.prob_df['Sw'])],
                hoverinfo='text', name='Most Likely Fluid'
            ),
            row=2, col=2
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="black", row=2, col=2)
        
        # Average probabilities bar chart
        fluid_cols = ['gas', 'oil', 'brine', 'shale']
        means = [self.prob_df[f'P_{f}'].mean() for f in fluid_cols if f'P_{f}' in self.prob_df.columns]
        available_cols = [f for f in fluid_cols if f'P_{f}' in self.prob_df.columns]
        colors_bar = ['red', 'green', 'blue', 'gray'][:len(available_cols)]
        
        fig.add_trace(
            go.Bar(
                x=[f.capitalize() for f in available_cols],
                y=means,
                marker_color=colors_bar,
                text=[f'{m:.3f}' for m in means],
                textposition='outside',
                name='Average Probability'
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=900, width=1400, hovermode='y unified')
        
        for i in range(1, 4):
            fig.update_yaxes(title_text="Depth (m)", row=1, col=i, autorange="reversed")
            fig.update_yaxes(title_text="Depth (m)", row=2, col=1, autorange="reversed")
            fig.update_yaxes(title_text="Depth (m)", row=2, col=2, autorange="reversed")
        
        fig.update_xaxes(title_text="Probability", row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=1, col=2, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=1, col=3, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=2, col=1, range=[0, 1])
        fig.update_xaxes(title_text="Probability", row=2, col=2, range=[0, 1])
        fig.update_xaxes(title_text="Fluid Type", row=2, col=3)
        
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics"""
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
        
        for fluid in ['gas', 'oil', 'brine', 'shale']:
            col = f'P_{fluid}'
            if col in self.prob_df.columns:
                stats_dict[f'prob_{fluid}'] = self.prob_df[col].mean()
        
        if 'Most_Likely' in self.prob_df.columns:
            stats_dict['most_likely'] = self.prob_df['Most_Likely'].value_counts().to_dict()
        
        return stats_dict


def generate_sample_data():
    """Generate synthetic well log data"""
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
    st.markdown('<div class="main-header">🛢️ AVO Fluid Inversion (AFI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on: FD Field, Niger Delta - Data-Driven Fluid Clusters from Well Logs</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        
        uploaded_file = st.file_uploader("Upload CSV file with well logs", type=['csv'])
        
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
        st.markdown("## 📊 Methodology")
        st.markdown("""
        **Data-Driven Fluid Clusters:**
        1. Extract dry rock moduli from well logs
        2. Apply Biot-Gassmann fluid substitution
        3. Generate synthetic AVO responses for brine, oil, gas
        4. Bayesian probability classification
        
        **Fluid Properties:**
        - **Brine**: K=2.5 GPa, ρ=1040 kg/m³
        - **Oil**: K=1.2 GPa, ρ=800 kg/m³  
        - **Gas**: K=0.1 GPa, ρ=200 kg/m³
        """)
    
    if 'df' in locals():
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(10))
            required_cols = ['DEPTH', 'Vp', 'Vs', 'Phie', 'GR', 'rho', 'RT', 'Sw', 'Vclay']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠ Missing columns: {missing_cols}")
        
        with st.spinner("Running AVO Fluid Inversion analysis..."):
            try:
                afi = AVOFluidInversionApp(df, depth_min, depth_max)
                stats = afi.get_summary_stats()
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Samples", stats['samples'])
                with col2: st.metric("Avg Porosity", f"{stats['avg_phi']:.3f}")
                with col3: st.metric("Avg Sw", f"{stats['avg_sw']:.3f}")
                with col4: st.metric("Avg Vclay", f"{stats['avg_vclay']:.3f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Gas Probability", f"{stats.get('prob_gas', 0):.3f}")
                with col2: st.metric("Oil Probability", f"{stats.get('prob_oil', 0):.3f}")
                with col3: st.metric("Brine Probability", f"{stats.get('prob_brine', 0):.3f}")
                with col4: st.metric("Shale Probability", f"{stats.get('prob_shale', 0):.3f}")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["📈 AVO Crossplot", "📊 Well Logs", "🎯 Probability Maps"])
                
                with tab1:
                    st.plotly_chart(afi.plot_avo_crossplot(), use_container_width=True)
                    with st.expander("ℹ️ Interpretation"):
                        st.markdown("""
                        - **Left Plot**: Fluid clusters generated from rock physics and fluid substitution using your log data
                        - **Right Plot**: Bayesian classification of actual well data into fluid types
                        - **Colors**: 🔴 Gas, 🟢 Oil, 🔵 Brine, ⚪ Shale
                        - Points colored by water saturation (dark blue = hydrocarbons)
                        """)
                
                with tab2:
                    st.plotly_chart(afi.plot_well_logs_continuous(), use_container_width=True)
                
                with tab3:
                    st.plotly_chart(afi.plot_probability_maps(), use_container_width=True)
                
                # Summary
                with st.expander("📊 Detailed Summary"):
                    st.markdown("### Fluid Cluster Centers")
                    for fluid, res in afi.fluid_clusters.items():
                        st.write(f"**{fluid}**: B={res['center'][0]:.3f}, A={res['center'][1]:.3f}, n={res['size']}")
                    
                    st.markdown("### Bayesian Classification Results")
                    if 'most_likely' in stats:
                        for fluid, count in stats['most_likely'].items():
                            st.write(f"- **{fluid.capitalize()}**: {count} samples ({count/stats['samples']*100:.1f}%)")
                    
                    st.markdown("### AVO Classification")
                    mean_A, mean_B = stats['avg_intercept'], stats['avg_gradient']
                    if mean_A < 0 and mean_B < -0.5:
                        st.success(f"**Class III/IV**: Strong AVO anomaly (mean A={mean_A:.3f}, B={mean_B:.3f})")
                    elif mean_A < 0 and mean_B < 0:
                        st.info(f"**Class II/III**: Moderate AVO anomaly (mean A={mean_A:.3f}, B={mean_B:.3f})")
                    else:
                        st.write(f"AVO Class: A={mean_A:.3f}, B={mean_B:.3f}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>AVO Fluid Inversion (AFI) - Data-Driven Clusters from Well Logs</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
