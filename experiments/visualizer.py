"""
Advanced Visualization Tools for Multi-Agent Bot Comparison
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json
from typing import Dict, List, Optional
from datetime import datetime

class BotPerformanceVisualizer:
    """Create comprehensive visualizations for bot comparison"""
    
    def __init__(self):
        self.transaction_df = None
        self.results_dict = None
    
    def load_transaction_data(self, filename: str):
        """Load transaction data from CSV"""
        self.transaction_df = pd.read_csv(filename)
        print(f"Loaded {len(self.transaction_df)} transactions")
    
    def load_results_data(self, filename: str):
        """Load results data from JSON"""
        with open(filename, 'r') as f:
            self.results_dict = json.load(f)
        print(f"Loaded results for {len(self.results_dict.get('bot_performance', {}))} bots")
    
    def create_roi_comparison(self) -> go.Figure:
        """Create ROI comparison chart"""
        if self.transaction_df is None:
            raise ValueError("No transaction data loaded")
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, bot_type in enumerate(self.transaction_df['bot_type'].unique()):
            bot_data = self.transaction_df[self.transaction_df['bot_type'] == bot_type]
            
            fig.add_trace(go.Scatter(
                x=bot_data['step'],
                y=bot_data['roi'],
                name=bot_type.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=3),
                mode='lines'
            ))
        
        fig.update_layout(
            title="Bot ROI Comparison Over Time",
            xaxis_title="Time Steps",
            yaxis_title="ROI (%)",
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard"""
        if self.transaction_df is None:
            raise ValueError("No transaction data loaded")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "ROI Over Time", "Capital Growth",
                "Rebalance Activity", "Gas vs Fees"
            ]
        )
        
        colors = {
            'adaptive_bollinger': '#1f77b4',
            'rl_rebalancer': '#ff7f0e', 
            'gas_aware': '#2ca02c',
            'baseline': '#d62728'
        }
        
        # 1. ROI Over Time
        for bot_type in self.transaction_df['bot_type'].unique():
            bot_data = self.transaction_df[self.transaction_df['bot_type'] == bot_type]
            fig.add_trace(
                go.Scatter(
                    x=bot_data['step'],
                    y=bot_data['roi'],
                    name=f"{bot_type}",
                    line=dict(color=colors.get(bot_type, '#000000')),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # 2. Capital Growth
        for bot_type in self.transaction_df['bot_type'].unique():
            bot_data = self.transaction_df[self.transaction_df['bot_type'] == bot_type]
            fig.add_trace(
                go.Scatter(
                    x=bot_data['step'],
                    y=bot_data['capital'],
                    name=f"{bot_type}",
                    line=dict(color=colors.get(bot_type, '#000000')),
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Rebalance counts
        rebalance_counts = self.transaction_df[
            self.transaction_df['action'].str.contains('rebalance')
        ].groupby('bot_type').size().reset_index(name='rebalances')
        
        fig.add_trace(
            go.Bar(
                x=rebalance_counts['bot_type'],
                y=rebalance_counts['rebalances'],
                name="Rebalances",
                marker_color=[colors.get(bt, '#000000') for bt in rebalance_counts['bot_type']],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Gas vs Fees
        for bot_type in self.transaction_df['bot_type'].unique():
            bot_data = self.transaction_df[self.transaction_df['bot_type'] == bot_type]
            cumulative_fees = bot_data['fees'].cumsum()
            cumulative_gas = bot_data['gas_cost'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_gas,
                    y=cumulative_fees,
                    name=f"{bot_type}",
                    mode='markers+lines',
                    marker=dict(color=colors.get(bot_type, '#000000')),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Multi-Agent Bot Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_summary_table(self) -> go.Figure:
        """Create summary performance table"""
        if self.results_dict is None:
            raise ValueError("No results data loaded")
        
        headers = ['Bot Type', 'Final ROI (%)', 'Rebalances', 'Net Profit']
        
        table_data = []
        for bot_id, perf in self.results_dict['bot_performance'].items():
            table_data.append([
                perf['bot_type'],
                f"{perf['final_roi']:.2f}%",
                str(perf['total_rebalances']),
                f"{perf['net_profit']:.6f}"
            ])
        
        # Sort by ROI
        table_data.sort(key=lambda x: float(x[1][:-1]), reverse=True)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='lightblue',
                align='center',
                font=dict(size=14)
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color='white',
                align='center',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Bot Performance Summary",
            height=400
        )
        
        return fig
    
    def save_all_charts(self, output_dir: str = "charts"):
        """Save all charts to HTML files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {
            'roi_comparison': self.create_roi_comparison(),
            'dashboard': self.create_performance_dashboard(),
            'summary': self.create_summary_table()
        }
        
        for name, fig in charts.items():
            filename = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filename)
            print(f"Saved {filename}")

def main():
    """Demo visualization"""
    viz = BotPerformanceVisualizer()
    print("Visualizer ready. Use viz.load_transaction_data() and viz.load_results_data() then viz.save_all_charts()")

if __name__ == "__main__":
    main() 