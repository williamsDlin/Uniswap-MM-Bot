"""
Streamlit Dashboard for Real-time Bot Performance Monitoring
Run with: streamlit run experiments/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time
import os
import glob
from datetime import datetime

# Try importing local modules
try:
    from runner import ExperimentRunner
    from visualizer import BotPerformanceVisualizer
except ImportError:
    st.error("Could not import runner or visualizer modules. Make sure you're running from the correct directory.")
    st.stop()

st.set_page_config(
    page_title="Multi-Agent Bot Dashboard",
    page_icon="🤖",
    layout="wide"
)

def main():
    """Main dashboard application"""
    st.title("🤖 Multi-Agent Uniswap Bot Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'experiment_data' not in st.session_state:
        st.session_state.experiment_data = None
    
    # Sidebar controls
    st.sidebar.title("⚙️ Experiment Controls")
    
    # Experiment settings
    st.sidebar.subheader("Settings")
    num_steps = st.sidebar.slider("Number of Steps", 100, 2000, 1000)
    
    # Bot selection
    st.sidebar.subheader("Bot Selection")
    bots = {
        "Adaptive Bollinger": "adaptive_bollinger",
        "RL Rebalancer": "rl_rebalancer",
        "Gas Aware": "gas_aware", 
        "Baseline": "baseline"
    }
    
    selected_bots = []
    for name, bot_type in bots.items():
        if st.sidebar.checkbox(name, value=True):
            selected_bots.append((name, bot_type))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Run Experiment", "📊 Results", "📈 Analysis"])
    
    with tab1:
        run_experiment_tab(selected_bots, num_steps)
    
    with tab2:
        results_tab()
    
    with tab3:
        analysis_tab()

def run_experiment_tab(selected_bots, num_steps):
    """Tab for running new experiments"""
    st.header("🚀 Run New Experiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.write(f"**Steps:** {num_steps}")
        st.write(f"**Selected Bots:** {len(selected_bots)}")
        
        for name, _ in selected_bots:
            st.write(f"• {name}")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🚀 Start Experiment", type="primary"):
            run_experiment(selected_bots, num_steps)
        
        if st.button("🗑️ Clear Results"):
            st.session_state.experiment_data = None
            st.success("Results cleared!")

def run_experiment(selected_bots, num_steps):
    """Run the experiment"""
    if not selected_bots:
        st.error("Please select at least one bot!")
        return
    
    try:
        with st.spinner("Running experiment..."):
            # Create runner
            runner = ExperimentRunner()
            
            # Add bots
            for name, bot_type in selected_bots:
                bot_id = name.lower().replace(' ', '_')
                runner.add_bot(bot_id, bot_type)
            
            # Run experiment
            results = runner.run_experiment(num_steps)
            
            # Save files
            results_file = runner.save_results()
            transaction_file = runner.save_transaction_data()
            
            # Store in session state
            st.session_state.experiment_data = {
                'results': results,
                'results_file': results_file,
                'transaction_file': transaction_file
            }
            
            st.success(f"✅ Experiment completed! Generated {num_steps} steps for {len(selected_bots)} bots.")
            
    except Exception as e:
        st.error(f"❌ Experiment failed: {str(e)}")

def results_tab():
    """Tab for showing current results"""
    st.header("📊 Current Results")
    
    if st.session_state.experiment_data is None:
        st.info("🔍 No experiment data. Run an experiment first.")
        return
    
    results = st.session_state.experiment_data['results']
    
    # Performance metrics
    st.subheader("🎯 Bot Performance")
    
    cols = st.columns(len(results['bot_performance']))
    
    for i, (bot_id, perf) in enumerate(results['bot_performance'].items()):
        with cols[i]:
            st.metric(
                label=f"🤖 {bot_id}",
                value=f"{perf['final_roi']:.2f}%",
                delta=f"{perf['total_rebalances']} rebalances"
            )
    
    # Market summary
    st.subheader("📈 Market Summary")
    market = results['market_data']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Price", f"${market['initial_price']:.2f}")
    with col2:
        st.metric("Final Price", f"${market['final_price']:.2f}")
    with col3:
        st.metric("Price Change", f"{market['price_change']:.2f}%")
    with col4:
        st.metric("Avg Gas", f"{market['avg_gas_price']:.1f} gwei")
    
    # Quick visualization
    if st.session_state.experiment_data.get('transaction_file'):
        st.subheader("📈 ROI Comparison")
        
        try:
            viz = BotPerformanceVisualizer()
            viz.load_transaction_data(st.session_state.experiment_data['transaction_file'])
            fig = viz.create_roi_comparison()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")

def analysis_tab():
    """Tab for detailed analysis"""
    st.header("📈 Detailed Analysis")
    
    # File selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Transaction Data")
        transaction_files = glob.glob("transaction_data_*.csv")
        if transaction_files:
            selected_transaction = st.selectbox(
                "Select transaction file",
                transaction_files,
                index=len(transaction_files)-1
            )
        else:
            st.warning("No transaction files found")
            return
    
    with col2:
        st.subheader("📁 Results Data")
        results_files = glob.glob("experiment_results_*.json")
        if results_files:
            selected_results = st.selectbox(
                "Select results file",
                results_files,
                index=len(results_files)-1
            )
        else:
            st.warning("No results files found")
            return
    
    # Generate analysis
    if st.button("📊 Generate Full Analysis"):
        try:
            with st.spinner("Creating visualizations..."):
                viz = BotPerformanceVisualizer()
                viz.load_transaction_data(selected_transaction)
                viz.load_results_data(selected_results)
                
                # ROI comparison
                st.subheader("📈 ROI Over Time")
                fig1 = viz.create_roi_comparison()
                st.plotly_chart(fig1, use_container_width=True)
                
                # Performance dashboard
                st.subheader("📊 Performance Dashboard")
                fig2 = viz.create_performance_dashboard()
                st.plotly_chart(fig2, use_container_width=True)
                
                # Summary table
                st.subheader("📋 Summary Table")
                fig3 = viz.create_summary_table()
                st.plotly_chart(fig3, use_container_width=True)
                
                # Save charts
                if st.button("💾 Save Charts to HTML"):
                    viz.save_all_charts()
                    st.success("Charts saved to 'charts' directory!")
                    
        except Exception as e:
            st.error(f"Error in analysis: {e}")
    
    # File browser
    st.subheader("📁 Available Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Transaction Files:**")
        for file in sorted(transaction_files, reverse=True):
            file_size = os.path.getsize(file) / 1024
            st.write(f"• {file} ({file_size:.1f} KB)")
    
    with col2:
        st.write("**Results Files:**")
        for file in sorted(results_files, reverse=True):
            file_size = os.path.getsize(file) / 1024
            st.write(f"• {file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main() 