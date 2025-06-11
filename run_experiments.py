#!/usr/bin/env python3
"""
Multi-Agent Uniswap Bot Experiment Runner
==========================================

This script guides you through running comprehensive experiments with
the multi-agent bot framework, including:

1. Basic Performance Comparison
2. Parameter Sensitivity Analysis  
3. Market Condition Testing
4. Visualization Generation
5. Dashboard Launch

Usage:
    python run_experiments.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.getcwd())

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, description):
    """Print formatted step"""
    print(f"\n🔸 Step {step}: {description}")
    print("-" * 50)

def check_dependencies():
    """Check if required dependencies are installed"""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'plotly', 'streamlit', 
        'torch', 'scikit-learn', 'statsmodels'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def setup_environment():
    """Set up the experiment environment"""
    print_step(2, "Setting Up Environment")
    
    # Create necessary directories
    directories = ['experiments', 'charts', 'logs', 'models']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def run_basic_experiment():
    """Run basic performance comparison experiment"""
    print_step(3, "Running Basic Performance Comparison")
    
    try:
        from experiments.runner import ExperimentRunner
        
        # Create runner
        runner = ExperimentRunner()
        
        # Add all bot types
        bots = [
            ("adaptive_bollinger", "adaptive_bollinger"),
            ("rl_rebalancer", "rl_rebalancer"),
            ("gas_aware", "gas_aware"),
            ("baseline", "baseline")
        ]
        
        for bot_id, bot_type in bots:
            runner.add_bot(bot_id, bot_type)
            print(f"🤖 Added {bot_id}")
        
        print("\n🚀 Starting experiment (1000 steps)...")
        start_time = time.time()
        
        # Run experiment
        results = runner.run_experiment(1000)
        
        duration = time.time() - start_time
        print(f"✅ Experiment completed in {duration:.2f} seconds")
        
        # Save results
        results_file = runner.save_results()
        transaction_file = runner.save_transaction_data()
        
        print(f"💾 Results saved: {results_file}")
        print(f"💾 Transactions saved: {transaction_file}")
        
        # Print summary
        print("\n📊 EXPERIMENT SUMMARY:")
        for bot_id, perf in results['bot_performance'].items():
            print(f"  {bot_id:20} | ROI: {perf['final_roi']:6.2f}% | Rebalances: {perf['total_rebalances']:3d}")
        
        return results_file, transaction_file
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return None, None

def run_parameter_analysis():
    """Run parameter sensitivity analysis"""
    print_step(4, "Running Parameter Sensitivity Analysis")
    
    try:
        from experiments.runner import ExperimentRunner
        
        print("🔬 Testing different configurations...")
        
        # Test configurations
        configs = [
            (500, "Short-term (500 steps)"),
            (1000, "Medium-term (1000 steps)"),
            (2000, "Long-term (2000 steps)")
        ]
        
        results_summary = []
        
        for steps, description in configs:
            print(f"\n📈 Running {description}...")
            
            runner = ExperimentRunner()
            runner.add_bot("adaptive", "adaptive_bollinger")
            runner.add_bot("baseline", "baseline")
            
            results = runner.run_experiment(steps)
            
            # Extract key metrics
            adaptive_roi = results['bot_performance']['adaptive']['final_roi']
            baseline_roi = results['bot_performance']['baseline']['final_roi']
            
            results_summary.append({
                'config': description,
                'steps': steps,
                'adaptive_roi': adaptive_roi,
                'baseline_roi': baseline_roi,
                'improvement': adaptive_roi - baseline_roi
            })
            
            print(f"  Adaptive ROI: {adaptive_roi:.2f}%")
            print(f"  Baseline ROI: {baseline_roi:.2f}%")
            print(f"  Improvement: {adaptive_roi - baseline_roi:.2f}%")
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"parameter_analysis_{timestamp}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Parameter analysis saved: {analysis_file}")
        
        return analysis_file
        
    except Exception as e:
        print(f"❌ Parameter analysis failed: {e}")
        return None

def generate_visualizations(results_file, transaction_file):
    """Generate comprehensive visualizations"""
    print_step(5, "Generating Visualizations")
    
    if not results_file or not transaction_file:
        print("⚠️  No data files available. Skipping visualization.")
        return
    
    try:
        from experiments.visualizer import BotPerformanceVisualizer
        
        print("🎨 Creating visualizations...")
        
        # Create visualizer
        viz = BotPerformanceVisualizer()
        viz.load_transaction_data(transaction_file)
        viz.load_results_data(results_file)
        
        # Generate all charts
        viz.save_all_charts()
        
        print("✅ Visualizations saved to 'charts' directory")
        print("📁 Charts created:")
        
        charts_dir = "charts"
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                if file.endswith('.html'):
                    print(f"   • {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print_step(6, "Launching Interactive Dashboard")
    
    try:
        print("🚀 Starting Streamlit dashboard...")
        print("📊 Dashboard will open in your browser")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        
        # Check if streamlit is available
        try:
            import streamlit
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit")
            return False
        
        # Launch dashboard
        dashboard_path = "experiments/dashboard.py"
        if os.path.exists(dashboard_path):
            subprocess.run(["streamlit", "run", dashboard_path])
        else:
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped")
        return True
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        return False

def run_quick_demo():
    """Run a quick demonstration"""
    print_step("Demo", "Quick Demonstration")
    
    try:
        from experiments.runner import ExperimentRunner
        
        runner = ExperimentRunner()
        runner.add_bot("adaptive", "adaptive_bollinger")
        runner.add_bot("baseline", "baseline")
        
        print("🚀 Running quick demo (100 steps)...")
        results = runner.run_experiment(100)
        
        print("\n📊 DEMO RESULTS:")
        for bot_id, perf in results['bot_performance'].items():
            print(f"  {bot_id}: {perf['final_roi']:.2f}% ROI")
        
        print("✅ Demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main experiment runner"""
    print_header("Multi-Agent Uniswap Bot Experiment Suite")
    print("🚀 Welcome to the comprehensive bot testing framework!")
    print("\nThis script will guide you through:")
    print("  1. Basic performance comparison")
    print("  2. Parameter sensitivity analysis")
    print("  3. Visualization generation")
    print("  4. Interactive dashboard launch")
    
    # Get user choice
    print("\n🔧 Choose your experiment:")
    print("  1. Full Experiment Suite (recommended)")
    print("  2. Quick Demo (fast)")
    print("  3. Just Launch Dashboard")
    print("  4. Parameter Analysis Only")
    print("  5. Generate Visualizations Only")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    if choice == "1":
        # Full experiment suite
        if not check_dependencies():
            return
        
        setup_environment()
        
        results_file, transaction_file = run_basic_experiment()
        
        if results_file and transaction_file:
            generate_visualizations(results_file, transaction_file)
            
            run_parameter_analysis()
            
            print_header("Experiment Complete!")
            print("✅ All experiments completed successfully!")
            print(f"📁 Results saved in current directory")
            print(f"📊 Charts saved in 'charts' directory")
            
            launch_choice = input("\n🚀 Launch interactive dashboard? (y/n): ").strip().lower()
            if launch_choice in ['y', 'yes']:
                launch_dashboard()
    
    elif choice == "2":
        # Quick demo
        setup_environment()
        run_quick_demo()
    
    elif choice == "3":
        # Just dashboard
        launch_dashboard()
    
    elif choice == "4":
        # Parameter analysis only
        setup_environment()
        run_parameter_analysis()
    
    elif choice == "5":
        # Visualizations only
        print("📁 Looking for existing data files...")
        
        import glob
        transaction_files = glob.glob("transaction_data_*.csv")
        results_files = glob.glob("experiment_results_*.json")
        
        if transaction_files and results_files:
            latest_transaction = sorted(transaction_files)[-1]
            latest_results = sorted(results_files)[-1]
            
            print(f"📊 Using: {latest_transaction}")
            print(f"📊 Using: {latest_results}")
            
            generate_visualizations(latest_results, latest_transaction)
        else:
            print("❌ No data files found. Run an experiment first.")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 