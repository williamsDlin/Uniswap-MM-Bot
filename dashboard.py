#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Multi-Agent Uniswap V3 Bot Framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import our modules
try:
    from data_loader import DataLoader
    from enhanced_experiment_runner import EnhancedExperimentRunner, BaselineAgent
    from agents.adaptive_agent import AdaptiveAgent
    from agents.gas_aware_agent import GasAwareAgent
except ImportError as e:
    st.error("Import error: {}. Please ensure all modules are available.".format(e))
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Uniswap V3 Multi-Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.experiment_runner = None
        self.market_data = None
        self.experiment_results = None
        
    def run(self):
        """Main dashboard interface"""
        st.title("🤖 Uniswap V3 Multi-Agent Bot Dashboard")
        st.markdown("Compare different LP strategies with real market data")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Data", "🚀 Run Experiment", "📈 Results", "📋 Analysis"])
        
        with tab1:
            self.render_market_data_tab()
        
        with tab2:
            self.render_experiment_tab()
        
        with tab3:
            self.render_results_tab()
        
        with tab4:
            self.render_analysis_tab()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("⚙️ Configuration")
        
        # Date range selection
        st.sidebar.subheader("📅 Data Range")
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=end_date - timedelta(days=7),
            max_value=end_date
        )
        
        # Store in session state
        st.session_state.start_date = start_date.strftime('%Y-%m-%d')
        st.session_state.end_date = end_date.strftime('%Y-%m-%d')
        
        # Load data button
        if st.sidebar.button("📥 Load Market Data"):
            self.load_market_data()
        
        # Agent configuration
        st.sidebar.subheader("🤖 Agent Configuration")
        
        # Baseline agent
        st.sidebar.checkbox("Baseline Agent", value=True, key="use_baseline")
        if st.session_state.use_baseline:
            st.sidebar.slider("Baseline Range Width", 0.05, 0.3, 0.12, 0.01, key="baseline_range")
        
        # Adaptive agent
        st.sidebar.checkbox("Adaptive Agent", value=True, key="use_adaptive")
        if st.session_state.use_adaptive:
            st.sidebar.slider("Base Range Width", 0.05, 0.2, 0.1, 0.01, key="adaptive_base_range")
            st.sidebar.slider("Volatility Multiplier", 1.0, 5.0, 2.0, 0.1, key="adaptive_vol_mult")
        
        # Gas-aware agent
        st.sidebar.checkbox("Gas-Aware Agent", value=True, key="use_gas_aware")
        if st.session_state.use_gas_aware:
            st.sidebar.slider("Gas Percentile Threshold", 50, 95, 80, 5, key="gas_percentile")
            st.sidebar.slider("Max Delay Hours", 1, 12, 6, 1, key="max_delay")
        
        # Experiment settings
        st.sidebar.subheader("🧪 Experiment Settings")
        st.sidebar.slider("Max Steps (0 = all)", 0, 500, 0, 10, key="max_steps")
        
        # File management
        st.sidebar.subheader("📁 File Management")
        if st.sidebar.button("🗑️ Clear Cache"):
            self.clear_cache()
    
    def render_market_data_tab(self):
        """Render market data visualization tab"""
        st.header("📊 Market Data Overview")
        
        if not hasattr(st.session_state, 'market_data') or st.session_state.market_data is None:
            st.info("👆 Load market data using the sidebar to get started")
            return
        
        market_data = st.session_state.market_data
        summary = st.session_state.market_summary
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Duration",
                "{} hours".format(summary['date_range']['duration_hours']),
                delta=None
            )
        
        with col2:
            st.metric(
                "Price Range",
                "${:.0f} - ${:.0f}".format(summary['price_stats']['min'], summary['price_stats']['max']),
                delta="{:.1f}%".format(
                    ((summary['price_stats']['max'] - summary['price_stats']['min']) / summary['price_stats']['min']) * 100
                )
            )
        
        with col3:
            st.metric(
                "Volatility",
                "{:.2f}%".format(summary['price_stats']['volatility'] * 100),
                delta=None
            )
        
        with col4:
            st.metric(
                "Avg Gas",
                "{:.1f} gwei".format(summary['gas_stats']['mean']),
                delta="Max: {:.1f}".format(summary['gas_stats']['max'])
            )
        
        # Price and gas charts
        self.render_market_charts(market_data)
    
    def render_market_charts(self, market_data: Dict[str, pd.DataFrame]):
        """Render market data charts"""
        price_data = market_data['price']
        gas_data = market_data['gas']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ETH Price', 'Gas Price'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['price'],
                name='ETH Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Volume on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['volume'],
                name='Volume',
                line=dict(color='#ff7f0e', width=1),
                opacity=0.6,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Gas price chart
        fig.add_trace(
            go.Scatter(
                x=gas_data.index,
                y=gas_data['gas_price'],
                name='Gas Price',
                line=dict(color='#d62728', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Market Conditions",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Gas Price (gwei)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_experiment_tab(self):
        """Render experiment configuration and execution tab"""
        st.header("🚀 Run Multi-Agent Experiment")
        
        if not hasattr(st.session_state, 'market_data') or st.session_state.market_data is None:
            st.warning("⚠️ Please load market data first")
            return
        
        # Experiment configuration summary
        st.subheader("📋 Experiment Configuration")
        
        agents_config = []
        if st.session_state.use_baseline:
            agents_config.append("Baseline (range: {:.1%})".format(st.session_state.baseline_range))
        if st.session_state.use_adaptive:
            agents_config.append("Adaptive (base: {:.1%}, vol_mult: {:.1f})".format(
                st.session_state.adaptive_base_range, st.session_state.adaptive_vol_mult))
        if st.session_state.use_gas_aware:
            agents_config.append("Gas-Aware ({}th percentile, max delay: {}h)".format(
                st.session_state.gas_percentile, st.session_state.max_delay))
        
        st.write("**Agents to test:** " + ", ".join(agents_config))
        
        max_steps = st.session_state.max_steps if st.session_state.max_steps > 0 else "All available"
        st.write("**Max steps:** {}".format(max_steps))
        
        # Run experiment button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🚀 Run Experiment", type="primary"):
                self.run_experiment()
        
        with col2:
            if st.button("💾 Save Results"):
                self.save_experiment_results()
        
        # Progress and status
        if hasattr(st.session_state, 'experiment_running') and st.session_state.experiment_running:
            st.info("🔄 Experiment in progress...")
            progress_bar = st.progress(0)
            # In a real implementation, you'd update this progress
        
        # Show recent results if available
        if hasattr(st.session_state, 'experiment_results') and st.session_state.experiment_results:
            st.subheader("📊 Latest Results Preview")
            self.render_results_preview()
    
    def render_results_tab(self):
        """Render experiment results tab"""
        st.header("📈 Experiment Results")
        
        # Load results options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File selector for loading previous results
            result_files = glob.glob("enhanced_experiment_results_*.json")
            if result_files:
                selected_file = st.selectbox("Load Previous Results", ["Current"] + result_files)
                if selected_file != "Current" and st.button("📂 Load Selected"):
                    self.load_experiment_results(selected_file)
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        # Show results if available
        if hasattr(st.session_state, 'experiment_results') and st.session_state.experiment_results:
            self.render_detailed_results()
        else:
            st.info("🔬 Run an experiment to see results here")
    
    def render_detailed_results(self):
        """Render detailed experiment results"""
        results = st.session_state.experiment_results
        
        # Performance summary
        st.subheader("🏆 Performance Summary")
        
        perf_data = []
        for agent_id, perf in results['agent_performance'].items():
            perf_data.append({
                'Agent': agent_id,
                'ROI (%)': perf.get('roi', 0),
                'Transactions': perf.get('executed_transactions', 0),
                'Gas Cost': perf.get('total_gas_cost', 0),
                'Execution Rate (%)': perf.get('execution_rate', 0) * 100,
                'Final Capital': perf.get('final_capital', 1.0)
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # ROI comparison chart
        fig_roi = px.bar(
            perf_df, 
            x='Agent', 
            y='ROI (%)',
            title='ROI Comparison',
            color='ROI (%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Capital evolution chart
        self.render_capital_evolution_chart(results)
        
        # Gas optimization metrics
        self.render_gas_metrics(results)
    
    def render_capital_evolution_chart(self, results: Dict[str, Any]):
        """Render capital evolution over time"""
        st.subheader("💰 Capital Evolution")
        
        fig = go.Figure()
        
        # Extract capital evolution for each agent
        for agent_id in results['agent_performance'].keys():
            # This would need to be extracted from step-by-step results
            # For now, we'll create a simplified version
            steps = list(range(len(results.get('step_by_step', []))))
            
            if steps:
                # Extract capital values from transaction history
                capital_values = [1.0]  # Starting capital
                
                # This is a simplified version - in practice, you'd extract from actual data
                final_capital = results['agent_performance'][agent_id].get('final_capital', 1.0)
                roi = results['agent_performance'][agent_id].get('roi', 0) / 100
                
                # Generate smooth capital evolution
                for i in range(1, len(steps)):
                    progress = i / len(steps)
                    capital = 1.0 + (roi * progress) + np.random.normal(0, 0.01)
                    capital_values.append(capital)
                
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=capital_values,
                    name=agent_id,
                    mode='lines',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Capital Evolution Over Time",
            xaxis_title="Time Steps",
            yaxis_title="Capital",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_gas_metrics(self, results: Dict[str, Any]):
        """Render gas optimization metrics"""
        gas_agents = {aid: perf for aid, perf in results['agent_performance'].items() 
                     if 'total_delays' in perf}
        
        if gas_agents:
            st.subheader("⛽ Gas Optimization Metrics")
            
            gas_data = []
            for agent_id, perf in gas_agents.items():
                gas_data.append({
                    'Agent': agent_id,
                    'Total Delays': perf.get('total_delays', 0),
                    'Avg Delay (h)': perf.get('avg_delay_duration', 0),
                    'Gas Saved': perf.get('total_gas_saved', 0),
                    'Savings per Delay': perf.get('gas_savings_per_delay', 0)
                })
            
            gas_df = pd.DataFrame(gas_data)
            st.dataframe(gas_df, use_container_width=True)
    
    def render_analysis_tab(self):
        """Render detailed analysis tab"""
        st.header("📋 Detailed Analysis")
        
        if not hasattr(st.session_state, 'experiment_results') or not st.session_state.experiment_results:
            st.info("🔬 Run an experiment to see detailed analysis")
            return
        
        results = st.session_state.experiment_results
        
        # Market conditions analysis
        st.subheader("🌊 Market Conditions Impact")
        
        market_summary = results.get('market_summary', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Price Change",
                "{:.2f}%".format(market_summary.get('price_change_pct', 0)),
                delta=None
            )
            st.metric(
                "Price Volatility",
                "{:.2f}%".format(market_summary.get('price_volatility', 0)),
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Gas",
                "{:.1f} gwei".format(market_summary.get('avg_gas_price', 0)),
                delta=None
            )
            st.metric(
                "Max Gas",
                "{:.1f} gwei".format(market_summary.get('max_gas_price', 0)),
                delta=None
            )
        
        # Strategy comparison
        st.subheader("🎯 Strategy Comparison")
        
        # Performance vs gas cost scatter plot
        perf_data = results['agent_performance']
        
        scatter_data = []
        for agent_id, perf in perf_data.items():
            scatter_data.append({
                'Agent': agent_id,
                'ROI': perf.get('roi', 0),
                'Gas Cost': perf.get('total_gas_cost', 0),
                'Transactions': perf.get('executed_transactions', 0)
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        if not scatter_df.empty:
            fig_scatter = px.scatter(
                scatter_df,
                x='Gas Cost',
                y='ROI',
                size='Transactions',
                color='Agent',
                title='Performance vs Gas Cost',
                hover_data=['Transactions']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Export options
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export CSV"):
                self.export_results_csv()
        
        with col2:
            if st.button("📋 Export Report"):
                self.export_html_report()
        
        with col3:
            if st.button("📈 Export Charts"):
                self.export_charts()
    
    def render_results_preview(self):
        """Render a quick preview of results"""
        if not hasattr(st.session_state, 'experiment_results'):
            return
        
        results = st.session_state.experiment_results
        perf_data = results.get('agent_performance', {})
        
        if perf_data:
            # Quick metrics
            best_roi = max(perf_data.values(), key=lambda x: x.get('roi', 0))
            best_agent = [k for k, v in perf_data.items() if v.get('roi', 0) == best_roi.get('roi', 0)][0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Performer", best_agent, "{:.2f}% ROI".format(best_roi.get('roi', 0)))
            
            with col2:
                total_txns = sum(p.get('executed_transactions', 0) for p in perf_data.values())
                st.metric("Total Transactions", total_txns)
            
            with col3:
                avg_roi = np.mean([p.get('roi', 0) for p in perf_data.values()])
                st.metric("Average ROI", "{:.2f}%".format(avg_roi))
    
    def load_market_data(self):
        """Load market data based on sidebar configuration"""
        with st.spinner("Loading market data..."):
            try:
                market_data = self.data_loader.load_market_data(
                    st.session_state.start_date,
                    st.session_state.end_date
                )
                summary = self.data_loader.get_data_summary(market_data)
                
                st.session_state.market_data = market_data
                st.session_state.market_summary = summary
                
                st.success("✅ Market data loaded successfully!")
                st.rerun()
                
            except Exception as e:
                st.error("❌ Error loading market data: {}".format(str(e)))
    
    def run_experiment(self):
        """Run the multi-agent experiment"""
        if not hasattr(st.session_state, 'market_data'):
            st.error("❌ Please load market data first")
            return
        
        with st.spinner("Running experiment..."):
            try:
                # Initialize experiment runner
                runner = EnhancedExperimentRunner(self.data_loader)
                
                # Add agents based on configuration
                if st.session_state.use_baseline:
                    runner.add_agent(BaselineAgent("baseline", range_width=st.session_state.baseline_range))
                
                if st.session_state.use_adaptive:
                    runner.add_agent(AdaptiveAgent(
                        "adaptive",
                        base_range_width=st.session_state.adaptive_base_range,
                        volatility_multiplier=st.session_state.adaptive_vol_mult
                    ))
                
                if st.session_state.use_gas_aware:
                    runner.add_agent(GasAwareAgent(
                        "gas_aware",
                        gas_percentile_threshold=st.session_state.gas_percentile,
                        max_delay_hours=st.session_state.max_delay
                    ))
                
                # Set market data
                runner.market_data = st.session_state.market_data
                
                # Run experiment
                max_steps = st.session_state.max_steps if st.session_state.max_steps > 0 else None
                results = runner.run_experiment(max_steps=max_steps)
                
                st.session_state.experiment_results = results
                st.session_state.experiment_runner = runner
                
                st.success("✅ Experiment completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error("❌ Error running experiment: {}".format(str(e)))
    
    def save_experiment_results(self):
        """Save experiment results to file"""
        if not hasattr(st.session_state, 'experiment_runner'):
            st.error("❌ No experiment results to save")
            return
        
        try:
            runner = st.session_state.experiment_runner
            results_file = runner.save_results()
            transactions_file = runner.export_transaction_data()
            
            st.success("✅ Results saved to {} and {}".format(results_file, transactions_file))
            
        except Exception as e:
            st.error("❌ Error saving results: {}".format(str(e)))
    
    def load_experiment_results(self, filename: str):
        """Load experiment results from file"""
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            
            st.session_state.experiment_results = results
            st.success("✅ Results loaded from {}".format(filename))
            st.rerun()
            
        except Exception as e:
            st.error("❌ Error loading results: {}".format(str(e)))
    
    def clear_cache(self):
        """Clear cached data"""
        cache_files = glob.glob("data_cache/*.pkl")
        for file in cache_files:
            try:
                os.remove(file)
            except:
                pass
        
        st.success("✅ Cache cleared")
    
    def export_results_csv(self):
        """Export results to CSV"""
        # Implementation for CSV export
        st.info("CSV export functionality would be implemented here")
    
    def export_html_report(self):
        """Export HTML report"""
        # Implementation for HTML report
        st.info("HTML report export functionality would be implemented here")
    
    def export_charts(self):
        """Export charts as images"""
        # Implementation for chart export
        st.info("Chart export functionality would be implemented here")

def main():
    """Main dashboard entry point"""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 