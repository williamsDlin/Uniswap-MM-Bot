"""
Multi-Bot Evaluation Framework
Compares performance of all AI-enhanced Uniswap V3 bots
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class BotEvaluator:
    """Comprehensive evaluation framework for multi-agent bot comparison"""
    
    def __init__(self, evaluation_period_hours: int = 168):
        """Initialize bot evaluator"""
        self.evaluation_period_hours = evaluation_period_hours
        self.bot_data = {}
        self.results = {}
        
    def load_bot_data(self, bot_id: str, csv_file: str, bot_type: str) -> None:
        """Load data from a bot's CSV log file"""
        try:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.bot_data[bot_id] = {
                'type': bot_type,
                'data': df,
                'total_actions': len(df)
            }
            
            logger.info(f"Loaded data for {bot_id}: {len(df)} actions")
            
        except Exception as e:
            logger.error(f"Error loading data for {bot_id}: {e}")
    
    def calculate_bot_metrics(self, bot_id: str) -> Dict[str, float]:
        """Calculate all metrics for a specific bot"""
        if bot_id not in self.bot_data:
            return {}
        
        df = self.bot_data[bot_id]['data']
        metrics = {}
        
        try:
            # Basic metrics
            total_cycles = df['cycle'].max() if 'cycle' in df.columns else len(df)
            rebalances = len(df[df['action'].str.contains('rebalance', case=False, na=False)])
            
            metrics['total_cycles'] = total_cycles
            metrics['total_rebalances'] = rebalances
            metrics['rebalance_frequency'] = rebalances / total_cycles if total_cycles > 0 else 0
            
            # Fees and costs
            fees_earned = df['fees_earned'].fillna(0).sum() if 'fees_earned' in df.columns else 0.001
            gas_cost = rebalances * 0.01  # Estimate
            
            metrics['lp_fees_earned'] = fees_earned
            metrics['total_gas_cost'] = gas_cost
            metrics['roi'] = ((fees_earned - gas_cost) / 1.0) * 100  # Assume 1 ETH deployed
            
            # Position effectiveness
            successful = len(df[df['status'] == 'success']) if 'status' in df.columns else rebalances
            failed = len(df[df['status'].isin(['failed', 'error'])]) if 'status' in df.columns else 0
            
            metrics['successful_positions'] = successful
            metrics['failed_positions'] = failed
            metrics['position_effectiveness'] = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 100
            
            # Adaptivity score based on bot type
            bot_type = self.bot_data[bot_id]['type']
            if bot_type == 'adaptive_bollinger':
                metrics['adaptivity_score'] = 80  # High for ML-based
            elif bot_type == 'rl_rebalancer':
                metrics['adaptivity_score'] = 90  # Highest for RL
            elif bot_type == 'gas_aware':
                metrics['adaptivity_score'] = 70  # High for gas optimization
            else:
                metrics['adaptivity_score'] = 50  # Medium for static
            
            logger.info(f"Calculated metrics for {bot_id}: ROI={metrics['roi']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {bot_id}: {e}")
        
        return metrics
    
    def compare_all_bots(self) -> Dict[str, Any]:
        """Compare all loaded bots and generate results"""
        if not self.bot_data:
            return {}
        
        all_metrics = {}
        for bot_id in self.bot_data.keys():
            all_metrics[bot_id] = self.calculate_bot_metrics(bot_id)
        
        # Create comparison
        comparison_df = pd.DataFrame(all_metrics).T
        
        # Generate rankings
        leaderboard = {}
        if 'roi' in comparison_df.columns:
            roi_ranking = comparison_df.sort_values('roi', ascending=False)
            leaderboard['roi_ranking'] = roi_ranking['roi'].to_dict()
        
        self.results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'bots_evaluated': list(self.bot_data.keys()),
            'metrics_table': comparison_df.to_dict('index'),
            'leaderboard': leaderboard
        }
        
        return self.results
    
    def save_results(self, output_file: str = "evaluation_results.json") -> None:
        """Save results to JSON"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Example usage"""
    evaluator = BotEvaluator()
    print("Bot evaluation framework ready")

if __name__ == "__main__":
    main() 