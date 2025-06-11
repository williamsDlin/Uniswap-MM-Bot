"""
Multi-Bot Evaluation Framework
Compares performance of all AI-enhanced Uniswap V3 bots
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

class BotEvaluator:
    """Comprehensive evaluation framework for multi-agent bot comparison"""
    
    def __init__(self, evaluation_period_hours: int = 168):  # 7 days = 168 hours
        """
        Initialize bot evaluator
        
        Args:
            evaluation_period_hours: Length of evaluation period in hours
        """
        self.evaluation_period_hours = evaluation_period_hours
        self.bot_data = {}
        self.price_data = []
        self.gas_data = []
        
        # Evaluation metrics
        self.metrics = [
            'strategy_decisions',
            'lp_fees_earned', 
            'gas_cost',
            'roi',
            'downtime_hours',
            'position_effectiveness',
            'adaptivity_score'
        ]
        
        # Results storage
        self.results = {}
        self.leaderboard = {}
        
    def load_bot_data(self, bot_id: str, csv_file: str, bot_type: str) -> None:
        """
        Load data from a bot's CSV log file
        
        Args:
            bot_id: Unique identifier for the bot
            csv_file: Path to bot's CSV log file
            bot_type: Type of bot ('adaptive_bollinger', 'rl_rebalancer', 'crosschain_arbitrage', 'gas_aware')
        """
        try:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.bot_data[bot_id] = {
                'type': bot_type,
                'data': df,
                'start_time': df['timestamp'].min(),
                'end_time': df['timestamp'].max(),
                'total_actions': len(df)
            }
            
            logger.info(f"Loaded data for {bot_id}: {len(df)} actions from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
        except Exception as e:
            logger.error(f"Error loading data for {bot_id}: {e}")
    
    def load_market_data(self, price_csv: str, gas_csv: str = None) -> None:
        """
        Load market price and gas price data
        
        Args:
            price_csv: CSV file with price data (timestamp, price)
            gas_csv: CSV file with gas price data (timestamp, gas_price_gwei)
        """
        try:
            # Load price data
            price_df = pd.read_csv(price_csv)
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            self.price_data = price_df.to_dict('records')
            
            # Load gas data if provided
            if gas_csv:
                gas_df = pd.read_csv(gas_csv)
                gas_df['timestamp'] = pd.to_datetime(gas_df['timestamp'])
                self.gas_data = gas_df.to_dict('records')
            
            logger.info(f"Loaded market data: {len(self.price_data)} price points, {len(self.gas_data)} gas points")
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
    
    def calculate_bot_metrics(self, bot_id: str) -> Dict[str, float]:
        """
        Calculate all metrics for a specific bot
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Dictionary of calculated metrics
        """
        if bot_id not in self.bot_data:
            return {}
        
        bot_info = self.bot_data[bot_id]
        df = bot_info['data']
        
        metrics = {}
        
        try:
            # 1. Strategy Decisions
            total_cycles = df['cycle'].max() if 'cycle' in df.columns else len(df)
            rebalance_actions = len(df[df['action'].str.contains('rebalance', case=False, na=False)])
            metrics['total_cycles'] = total_cycles
            metrics['total_rebalances'] = rebalance_actions
            metrics['rebalance_frequency'] = rebalance_actions / total_cycles if total_cycles > 0 else 0
            
            # 2. LP Fees Earned
            if 'fees_earned' in df.columns:
                total_fees = df['fees_earned'].fillna(0).sum()
            else:
                # Estimate fees based on position time and typical 0.3% APY
                total_fees = 0.001  # Placeholder
            metrics['lp_fees_earned'] = total_fees
            
            # 3. Gas Cost
            if 'gas_used' in df.columns and 'current_gas_gwei' in df.columns:
                # Calculate actual gas costs
                gas_costs = []
                for _, row in df.iterrows():
                    if pd.notna(row.get('gas_used')) and pd.notna(row.get('current_gas_gwei')):
                        gas_cost_eth = (row['gas_used'] * row['current_gas_gwei'] * 1e-9) / 1e18
                        gas_costs.append(gas_cost_eth)
                total_gas_cost = sum(gas_costs)
            else:
                # Estimate gas costs
                total_gas_cost = rebalance_actions * 0.01  # ~0.01 ETH per rebalance
            metrics['total_gas_cost'] = total_gas_cost
            
            # 4. ROI (Return on Investment)
            capital_deployed = 1.0  # Assume 1 ETH equivalent deployed
            net_profit = total_fees - total_gas_cost
            roi = (net_profit / capital_deployed) * 100  # Percentage
            metrics['roi'] = roi
            metrics['net_profit'] = net_profit
            
            # 5. Downtime (hours with no position active)
            # Estimate based on successful vs failed transactions
            successful_positions = len(df[df['status'] == 'success']) if 'status' in df.columns else rebalance_actions
            failed_positions = len(df[df['status'].isin(['failed', 'error'])]) if 'status' in df.columns else 0
            
            # Assume each failed position = 1 hour downtime
            downtime_hours = failed_positions * 1.0
            metrics['downtime_hours'] = downtime_hours
            metrics['successful_positions'] = successful_positions
            metrics['failed_positions'] = failed_positions
            
            # 6. Position Effectiveness
            if len(self.price_data) > 0 and 'current_price' in df.columns:
                # Calculate how often price was within position ranges
                in_range_count = 0
                total_position_periods = 0
                
                for _, row in df.iterrows():
                    if pd.notna(row.get('lower_range')) and pd.notna(row.get('upper_range')):
                        current_price = row.get('current_price', 0)
                        if row['lower_range'] <= current_price <= row['upper_range']:
                            in_range_count += 1
                        total_position_periods += 1
                
                position_effectiveness = (in_range_count / total_position_periods * 100) if total_position_periods > 0 else 0
            else:
                position_effectiveness = 75.0  # Placeholder
            
            metrics['position_effectiveness'] = position_effectiveness
            
            # 7. Adaptivity Score (for ML bots)
            adaptivity_score = 0.0
            if bot_info['type'] == 'adaptive_bollinger':
                # Score based on parameter changes
                if 'window_size' in df.columns and 'std_dev' in df.columns:
                    window_changes = df['window_size'].nunique()
                    std_changes = df['std_dev'].nunique()
                    adaptivity_score = min(100, (window_changes + std_changes) * 5)
                else:
                    adaptivity_score = 50  # Default for adaptive bots
                    
            elif bot_info['type'] == 'rl_rebalancer':
                # Score based on action diversity
                if 'rl_action' in df.columns:
                    action_diversity = df['rl_action'].nunique()
                    adaptivity_score = min(100, action_diversity * 20)
                else:
                    adaptivity_score = 60  # Default for RL bots
                    
            elif bot_info['type'] == 'gas_aware':
                # Score based on gas optimization decisions
                delayed_txs = len(df[df['action'].str.contains('delay', case=False, na=False)])
                executed_txs = len(df[df['action'].str.contains('executed', case=False, na=False)])
                if executed_txs > 0:
                    adaptivity_score = min(100, (delayed_txs / executed_txs) * 100)
                else:
                    adaptivity_score = 30
            else:
                adaptivity_score = 20  # Static strategy
            
            metrics['adaptivity_score'] = adaptivity_score
            
            # 8. Additional bot-specific metrics
            if bot_info['type'] == 'adaptive_bollinger':
                # Bandit learning metrics
                if 'bandit_confidence' in df.columns:
                    avg_confidence = df['bandit_confidence'].fillna(0).mean()
                    metrics['avg_bandit_confidence'] = avg_confidence
                
            elif bot_info['type'] == 'rl_rebalancer':
                # RL performance metrics
                if 'reward' in df.columns:
                    total_reward = df['reward'].fillna(0).sum()
                    avg_reward = df['reward'].fillna(0).mean()
                    metrics['total_rl_reward'] = total_reward
                    metrics['avg_rl_reward'] = avg_reward
                
            elif bot_info['type'] == 'gas_aware':
                # Gas optimization metrics
                if 'gas_savings_percent' in df.columns:
                    avg_gas_savings = df['gas_savings_percent'].fillna(0).mean()
                    metrics['avg_gas_savings'] = avg_gas_savings
            
            logger.info(f"Calculated metrics for {bot_id}: ROI={roi:.2f}%, Gas={total_gas_cost:.4f} ETH")
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {bot_id}: {e}")
            return {}
        
        return metrics
    
    def compare_all_bots(self) -> Dict[str, Any]:
        """
        Compare all loaded bots and generate comprehensive results
        
        Returns:
            Complete comparison results
        """
        if not self.bot_data:
            logger.error("No bot data loaded")
            return {}
        
        # Calculate metrics for all bots
        all_metrics = {}
        for bot_id in self.bot_data.keys():
            all_metrics[bot_id] = self.calculate_bot_metrics(bot_id)
        
        # Create comparison table
        comparison_df = pd.DataFrame(all_metrics).T
        
        # Generate leaderboard
        leaderboard = {}
        
        # Overall ranking based on ROI
        if 'roi' in comparison_df.columns:
            roi_ranking = comparison_df.sort_values('roi', ascending=False)
            leaderboard['roi_ranking'] = roi_ranking[['roi', 'net_profit', 'total_gas_cost']].to_dict('index')
        
        # Gas efficiency ranking
        if 'total_gas_cost' in comparison_df.columns:
            gas_ranking = comparison_df.sort_values('total_gas_cost', ascending=True)
            leaderboard['gas_efficiency_ranking'] = gas_ranking[['total_gas_cost', 'total_rebalances']].to_dict('index')
        
        # Adaptivity ranking
        if 'adaptivity_score' in comparison_df.columns:
            adaptivity_ranking = comparison_df.sort_values('adaptivity_score', ascending=False)
            leaderboard['adaptivity_ranking'] = adaptivity_ranking[['adaptivity_score', 'position_effectiveness']].to_dict('index')
        
        # Overall score (weighted combination)
        if all(col in comparison_df.columns for col in ['roi', 'position_effectiveness', 'adaptivity_score']):
            comparison_df['overall_score'] = (
                comparison_df['roi'] * 0.4 +  # 40% weight on ROI
                comparison_df['position_effectiveness'] * 0.3 +  # 30% weight on effectiveness
                comparison_df['adaptivity_score'] * 0.2 +  # 20% weight on adaptivity
                (100 - comparison_df['downtime_hours']) * 0.1  # 10% weight on uptime
            )
            
            overall_ranking = comparison_df.sort_values('overall_score', ascending=False)
            leaderboard['overall_ranking'] = overall_ranking[['overall_score', 'roi', 'position_effectiveness', 'adaptivity_score']].to_dict('index')
        
        # Store results
        self.results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_period_hours': self.evaluation_period_hours,
            'bots_evaluated': list(self.bot_data.keys()),
            'metrics_table': comparison_df.to_dict('index'),
            'leaderboard': leaderboard,
            'summary_stats': {
                'total_bots': len(self.bot_data),
                'best_roi_bot': comparison_df['roi'].idxmax() if 'roi' in comparison_df.columns else None,
                'most_efficient_gas_bot': comparison_df['total_gas_cost'].idxmin() if 'total_gas_cost' in comparison_df.columns else None,
                'most_adaptive_bot': comparison_df['adaptivity_score'].idxmax() if 'adaptivity_score' in comparison_df.columns else None
            }
        }
        
        self.leaderboard = leaderboard
        
        logger.info(f"Comparison completed for {len(self.bot_data)} bots")
        return self.results
    
    def save_results(self, output_file: str = "evaluation_results.json") -> None:
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def create_leaderboard_json(self, output_file: str = "leaderboard.json") -> None:
        """Create leaderboard JSON for dashboard"""
        try:
            leaderboard_data = {
                'last_updated': datetime.now().isoformat(),
                'evaluation_period': f"{self.evaluation_period_hours} hours",
                'rankings': self.leaderboard
            }
            
            with open(output_file, 'w') as f:
                json.dump(leaderboard_data, f, indent=2, default=str)
            logger.info(f"Leaderboard saved to {output_file}")
        except Exception as e:
            logger.error(f"Error creating leaderboard: {e}")
    
    def generate_plots(self) -> Dict[str, go.Figure]:
        """Generate visualization plots for the evaluation"""
        plots = {}
        
        if not self.results:
            logger.error("No results to plot. Run compare_all_bots() first.")
            return plots
        
        try:
            # Get metrics data
            metrics_df = pd.DataFrame(self.results['metrics_table']).T
            
            # 1. ROI Comparison Bar Chart
            if 'roi' in metrics_df.columns:
                fig_roi = go.Figure(data=[
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df['roi'],
                        text=[f"{x:.2f}%" for x in metrics_df['roi']],
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])
                fig_roi.update_layout(
                    title="Bot Performance: Return on Investment (ROI)",
                    xaxis_title="Bot ID",
                    yaxis_title="ROI (%)",
                    showlegend=False
                )
                plots['roi_comparison'] = fig_roi
            
            # 2. Multi-metric Radar Chart
            if len(metrics_df) > 0:
                radar_metrics = ['roi', 'position_effectiveness', 'adaptivity_score']
                available_metrics = [m for m in radar_metrics if m in metrics_df.columns]
                
                if len(available_metrics) >= 3:
                    fig_radar = go.Figure()
                    
                    for bot_id in metrics_df.index:
                        values = [metrics_df.loc[bot_id, m] for m in available_metrics]
                        values.append(values[0])  # Close the radar chart
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=available_metrics + [available_metrics[0]],
                            fill='toself',
                            name=bot_id
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Multi-Metric Performance Comparison"
                    )
                    plots['radar_comparison'] = fig_radar
            
            # 3. Gas Efficiency vs ROI Scatter Plot
            if 'total_gas_cost' in metrics_df.columns and 'roi' in metrics_df.columns:
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=metrics_df['total_gas_cost'],
                    y=metrics_df['roi'],
                    mode='markers+text',
                    text=metrics_df.index,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=metrics_df['adaptivity_score'] if 'adaptivity_score' in metrics_df.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Adaptivity Score")
                    )
                ))
                
                fig_scatter.update_layout(
                    title="Gas Efficiency vs ROI",
                    xaxis_title="Total Gas Cost (ETH)",
                    yaxis_title="ROI (%)",
                    showlegend=False
                )
                plots['gas_vs_roi'] = fig_scatter
            
            # 4. Time Series Performance (if we have time-based data)
            for bot_id, bot_info in self.bot_data.items():
                df = bot_info['data']
                if 'timestamp' in df.columns and 'current_price' in df.columns:
                    fig_ts = go.Figure()
                    
                    # Plot price
                    fig_ts.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['current_price'],
                        mode='lines',
                        name='ETH/USDC Price',
                        line=dict(color='blue')
                    ))
                    
                    # Plot rebalance events
                    rebalance_df = df[df['action'].str.contains('rebalance', case=False, na=False)]
                    if len(rebalance_df) > 0:
                        fig_ts.add_trace(go.Scatter(
                            x=rebalance_df['timestamp'],
                            y=rebalance_df['current_price'],
                            mode='markers',
                            name='Rebalances',
                            marker=dict(color='red', size=8)
                        ))
                    
                    fig_ts.update_layout(
                        title=f"{bot_id} - Price and Actions Timeline",
                        xaxis_title="Time",
                        yaxis_title="Price / Action Markers",
                        showlegend=True
                    )
                    plots[f'{bot_id}_timeline'] = fig_ts
            
            logger.info(f"Generated {len(plots)} visualization plots")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plots

def main():
    """Example usage of the bot evaluator"""
    evaluator = BotEvaluator(evaluation_period_hours=168)  # 7 days
    
    # Load bot data (these files would be generated by running the bots)
    # evaluator.load_bot_data("bot_a", "bot_a_adaptive_actions_20241201_120000.csv", "adaptive_bollinger")
    # evaluator.load_bot_data("bot_b", "bot_b_rl_actions_20241201_120000.csv", "rl_rebalancer")
    # evaluator.load_bot_data("bot_c", "bot_c_crosschain_actions_20241201_120000.csv", "crosschain_arbitrage")
    # evaluator.load_bot_data("bot_d", "bot_d_gas_aware_actions_20241201_120000.csv", "gas_aware")
    
    # Load market data
    # evaluator.load_market_data("market_prices.csv", "gas_prices.csv")
    
    # Run evaluation
    # results = evaluator.compare_all_bots()
    
    # Save results
    # evaluator.save_results("evaluation_results.json")
    # evaluator.create_leaderboard_json("leaderboard.json")
    
    # Generate plots
    # plots = evaluator.generate_plots()
    
    print("Bot evaluation framework ready. Load data and run evaluation.")

if __name__ == "__main__":
    main() 