#!/usr/bin/env python3
"""
Performance Analysis Module for Multi-Agent Uniswap V3 Framework
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PerformanceEvaluator:
    """
    Comprehensive performance analysis for multi-agent experiments
    """
    
    def __init__(self, results_data: Optional[Dict[str, Any]] = None):
        self.results_data = results_data
        self.analysis_cache = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("Performance Evaluator initialized")
    
    def load_results(self, filename: str):
        """Load experiment results from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.results_data = json.load(f)
            print("Results loaded from: {}".format(filename))
        except Exception as e:
            print("Error loading results: {}".format(e))
            raise
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for all agents"""
        if not self.results_data:
            raise ValueError("No results data available. Load results first.")
        
        agent_performance = self.results_data['agent_performance']
        market_summary = self.results_data.get('market_summary', {})
        
        comprehensive_metrics = {
            'individual_metrics': {},
            'comparative_metrics': {},
            'market_adjusted_metrics': {},
            'risk_metrics': {},
            'efficiency_metrics': {}
        }
        
        # Individual agent metrics
        for agent_id, perf in agent_performance.items():
            individual = self._calculate_individual_metrics(agent_id, perf, market_summary)
            comprehensive_metrics['individual_metrics'][agent_id] = individual
        
        # Comparative metrics
        comprehensive_metrics['comparative_metrics'] = self._calculate_comparative_metrics(agent_performance)
        
        # Market-adjusted metrics
        comprehensive_metrics['market_adjusted_metrics'] = self._calculate_market_adjusted_metrics(
            agent_performance, market_summary)
        
        # Risk metrics
        comprehensive_metrics['risk_metrics'] = self._calculate_risk_metrics(agent_performance)
        
        # Efficiency metrics
        comprehensive_metrics['efficiency_metrics'] = self._calculate_efficiency_metrics(agent_performance)
        
        self.analysis_cache['comprehensive_metrics'] = comprehensive_metrics
        return comprehensive_metrics
    
    def _calculate_individual_metrics(self, agent_id: str, perf: Dict[str, Any], 
                                    market_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed metrics for individual agent"""
        roi = perf.get('roi', 0)
        total_transactions = perf.get('total_transactions', 0)
        executed_transactions = perf.get('executed_transactions', 0)
        total_gas_cost = perf.get('total_gas_cost', 0)
        execution_rate = perf.get('execution_rate', 0)
        
        # Basic performance
        metrics = {
            'roi': roi,
            'absolute_return': perf.get('net_profit', 0),
            'total_transactions': total_transactions,
            'executed_transactions': executed_transactions,
            'execution_rate': execution_rate,
            'total_gas_cost': total_gas_cost
        }
        
        # Risk-adjusted returns
        market_volatility = market_summary.get('price_volatility', 5.0) / 100  # Convert to decimal
        if market_volatility > 0:
            metrics['sharpe_ratio'] = roi / (market_volatility * 100)  # Simplified Sharpe ratio
            metrics['return_per_volatility'] = roi / market_volatility
        else:
            metrics['sharpe_ratio'] = 0
            metrics['return_per_volatility'] = 0
        
        # Efficiency metrics
        if executed_transactions > 0:
            metrics['roi_per_transaction'] = roi / executed_transactions
            metrics['gas_cost_per_transaction'] = total_gas_cost / executed_transactions
            metrics['net_return_per_transaction'] = (roi - total_gas_cost * 100) / executed_transactions
        else:
            metrics['roi_per_transaction'] = 0
            metrics['gas_cost_per_transaction'] = 0
            metrics['net_return_per_transaction'] = 0
        
        # Gas optimization metrics (if available)
        if 'total_delays' in perf:
            metrics['gas_optimization'] = {
                'total_delays': perf.get('total_delays', 0),
                'avg_delay_duration': perf.get('avg_delay_duration', 0),
                'total_gas_saved': perf.get('total_gas_saved', 0),
                'gas_savings_per_delay': perf.get('gas_savings_per_delay', 0)
            }
        
        # Position management metrics
        time_in_position = perf.get('time_in_position_ratio', 0)
        metrics['time_in_position_ratio'] = time_in_position
        metrics['position_utilization'] = time_in_position  # How much time agent was actively positioned
        
        return metrics
    
    def _calculate_comparative_metrics(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparative metrics across all agents"""
        rois = [perf.get('roi', 0) for perf in agent_performance.values()]
        gas_costs = [perf.get('total_gas_cost', 0) for perf in agent_performance.values()]
        execution_rates = [perf.get('execution_rate', 0) for perf in agent_performance.values()]
        
        return {
            'roi_ranking': self._rank_agents(agent_performance, 'roi'),
            'gas_efficiency_ranking': self._rank_agents(agent_performance, 'total_gas_cost', ascending=True),
            'execution_rate_ranking': self._rank_agents(agent_performance, 'execution_rate'),
            'roi_statistics': {
                'mean': np.mean(rois),
                'std': np.std(rois),
                'min': np.min(rois),
                'max': np.max(rois),
                'range': np.max(rois) - np.min(rois)
            },
            'gas_cost_statistics': {
                'mean': np.mean(gas_costs),
                'std': np.std(gas_costs),
                'min': np.min(gas_costs),
                'max': np.max(gas_costs)
            },
            'performance_spread': np.max(rois) - np.min(rois),
            'best_performer': max(agent_performance.keys(), key=lambda x: agent_performance[x].get('roi', 0)),
            'most_efficient': min(agent_performance.keys(), key=lambda x: agent_performance[x].get('total_gas_cost', float('inf')))
        }
    
    def _calculate_market_adjusted_metrics(self, agent_performance: Dict[str, Any], 
                                         market_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market-adjusted performance metrics"""
        market_return = market_summary.get('price_change_pct', 0)
        market_volatility = market_summary.get('price_volatility', 5.0)
        
        adjusted_metrics = {}
        
        for agent_id, perf in agent_performance.items():
            roi = perf.get('roi', 0)
            
            # Alpha (excess return over market)
            alpha = roi - market_return
            
            # Beta (sensitivity to market movements) - simplified
            beta = roi / market_return if market_return != 0 else 0
            
            # Information ratio
            tracking_error = abs(roi - market_return)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            adjusted_metrics[agent_id] = {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'market_adjusted_return': alpha,
                'tracking_error': tracking_error
            }
        
        return adjusted_metrics
    
    def _calculate_risk_metrics(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-related metrics"""
        risk_metrics = {}
        
        for agent_id, perf in agent_performance.items():
            roi = perf.get('roi', 0)
            execution_rate = perf.get('execution_rate', 0)
            
            # Risk score based on execution rate and volatility
            execution_risk = 1 - execution_rate  # Higher when execution rate is low
            
            # Downside risk (simplified)
            downside_risk = max(0, -roi)  # Only negative returns
            
            # Risk-adjusted return
            total_risk = execution_risk + downside_risk / 100
            risk_adjusted_return = roi / (1 + total_risk) if total_risk > 0 else roi
            
            risk_metrics[agent_id] = {
                'execution_risk': execution_risk,
                'downside_risk': downside_risk,
                'total_risk_score': total_risk,
                'risk_adjusted_return': risk_adjusted_return,
                'risk_return_ratio': roi / max(total_risk, 0.01)
            }
        
        return risk_metrics
    
    def _calculate_efficiency_metrics(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate efficiency-related metrics"""
        efficiency_metrics = {}
        
        for agent_id, perf in agent_performance.items():
            roi = perf.get('roi', 0)
            gas_cost = perf.get('total_gas_cost', 0)
            transactions = perf.get('executed_transactions', 0)
            
            # Gas efficiency
            gas_efficiency = roi / max(gas_cost, 0.001)  # ROI per unit gas cost
            
            # Transaction efficiency
            transaction_efficiency = roi / max(transactions, 1)  # ROI per transaction
            
            # Overall efficiency score
            efficiency_score = (gas_efficiency + transaction_efficiency) / 2
            
            efficiency_metrics[agent_id] = {
                'gas_efficiency': gas_efficiency,
                'transaction_efficiency': transaction_efficiency,
                'overall_efficiency': efficiency_score,
                'cost_effectiveness': roi - (gas_cost * 100)  # Net return after gas costs
            }
        
        return efficiency_metrics
    
    def _rank_agents(self, agent_performance: Dict[str, Any], metric: str, 
                    ascending: bool = False) -> List[Tuple[str, float]]:
        """Rank agents by a specific metric"""
        agent_scores = [(agent_id, perf.get(metric, 0)) for agent_id, perf in agent_performance.items()]
        return sorted(agent_scores, key=lambda x: x[1], reverse=not ascending)
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive performance report"""
        if not self.results_data:
            raise ValueError("No results data available")
        
        metrics = self.calculate_comprehensive_metrics()
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-AGENT UNISWAP V3 PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Experiment metadata
        metadata = self.results_data.get('experiment_metadata', {})
        report_lines.append("EXPERIMENT OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append("Total Steps: {}".format(metadata.get('total_steps', 'N/A')))
        report_lines.append("Agents Tested: {}".format(', '.join(metadata.get('agents', []))))
        report_lines.append("Start Time: {}".format(metadata.get('start_time', 'N/A')))
        report_lines.append("End Time: {}".format(metadata.get('end_time', 'N/A')))
        report_lines.append("")
        
        # Market conditions
        market_summary = self.results_data.get('market_summary', {})
        report_lines.append("MARKET CONDITIONS")
        report_lines.append("-" * 40)
        report_lines.append("Price Change: {:.2f}%".format(market_summary.get('price_change_pct', 0)))
        report_lines.append("Price Volatility: {:.2f}%".format(market_summary.get('price_volatility', 0)))
        report_lines.append("Average Gas Price: {:.1f} gwei".format(market_summary.get('avg_gas_price', 0)))
        report_lines.append("Max Gas Price: {:.1f} gwei".format(market_summary.get('max_gas_price', 0)))
        report_lines.append("")
        
        # Performance rankings
        comparative = metrics['comparative_metrics']
        report_lines.append("PERFORMANCE RANKINGS")
        report_lines.append("-" * 40)
        
        report_lines.append("ROI Ranking:")
        for i, (agent, roi) in enumerate(comparative['roi_ranking'], 1):
            report_lines.append("  {}. {}: {:.2f}%".format(i, agent, roi))
        
        report_lines.append("\nGas Efficiency Ranking:")
        for i, (agent, gas_cost) in enumerate(comparative['gas_efficiency_ranking'], 1):
            report_lines.append("  {}. {}: {:.4f} gas cost".format(i, agent, gas_cost))
        
        report_lines.append("")
        
        # Individual agent analysis
        report_lines.append("DETAILED AGENT ANALYSIS")
        report_lines.append("-" * 40)
        
        for agent_id, individual_metrics in metrics['individual_metrics'].items():
            report_lines.append("\n{} Analysis:".format(agent_id.upper()))
            report_lines.append("  ROI: {:.2f}%".format(individual_metrics['roi']))
            report_lines.append("  Sharpe Ratio: {:.2f}".format(individual_metrics['sharpe_ratio']))
            report_lines.append("  Execution Rate: {:.1f}%".format(individual_metrics['execution_rate'] * 100))
            report_lines.append("  Gas Cost per Transaction: {:.4f}".format(individual_metrics['gas_cost_per_transaction']))
            report_lines.append("  Time in Position: {:.1f}%".format(individual_metrics['time_in_position_ratio'] * 100))
            
            # Gas optimization metrics if available
            if 'gas_optimization' in individual_metrics:
                gas_opt = individual_metrics['gas_optimization']
                report_lines.append("  Gas Delays: {}".format(gas_opt['total_delays']))
                report_lines.append("  Gas Saved: {:.2f}".format(gas_opt['total_gas_saved']))
        
        # Risk analysis
        report_lines.append("\n\nRISK ANALYSIS")
        report_lines.append("-" * 40)
        
        risk_metrics = metrics['risk_metrics']
        for agent_id, risk in risk_metrics.items():
            report_lines.append("{}: Risk Score {:.3f}, Risk-Adjusted Return {:.2f}%".format(
                agent_id, risk['total_risk_score'], risk['risk_adjusted_return']))
        
        # Efficiency analysis
        report_lines.append("\n\nEFFICIENCY ANALYSIS")
        report_lines.append("-" * 40)
        
        efficiency_metrics = metrics['efficiency_metrics']
        for agent_id, eff in efficiency_metrics.items():
            report_lines.append("{}: Overall Efficiency {:.2f}, Cost Effectiveness {:.2f}%".format(
                agent_id, eff['overall_efficiency'], eff['cost_effectiveness']))
        
        # Key insights
        report_lines.append("\n\nKEY INSIGHTS")
        report_lines.append("-" * 40)
        
        best_performer = comparative['best_performer']
        most_efficient = comparative['most_efficient']
        performance_spread = comparative['performance_spread']
        
        report_lines.append("• Best Overall Performer: {}".format(best_performer))
        report_lines.append("• Most Gas Efficient: {}".format(most_efficient))
        report_lines.append("• Performance Spread: {:.2f}%".format(performance_spread))
        
        if performance_spread > 5:
            report_lines.append("• High performance variation suggests significant strategy differences")
        else:
            report_lines.append("• Low performance variation suggests similar strategy effectiveness")
        
        # Market impact analysis
        market_adjusted = metrics['market_adjusted_metrics']
        best_alpha = max(market_adjusted.values(), key=lambda x: x['alpha'])['alpha']
        if best_alpha > 2:
            report_lines.append("• Strong alpha generation indicates effective LP strategies")
        elif best_alpha < -2:
            report_lines.append("• Negative alpha suggests strategies underperformed market")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = "performance_report_{}.txt".format(timestamp)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print("Performance report saved to: {}".format(output_file))
        return report_text
    
    def create_performance_visualizations(self, output_dir: str = "performance_charts"):
        """Create comprehensive performance visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results_data:
            raise ValueError("No results data available")
        
        metrics = self.calculate_comprehensive_metrics()
        
        # 1. ROI Comparison Chart
        self._create_roi_comparison_chart(metrics, output_dir)
        
        # 2. Risk-Return Scatter Plot
        self._create_risk_return_scatter(metrics, output_dir)
        
        # 3. Efficiency Analysis
        self._create_efficiency_analysis(metrics, output_dir)
        
        # 4. Gas Optimization Analysis
        self._create_gas_optimization_chart(metrics, output_dir)
        
        # 5. Performance vs Market Conditions
        self._create_market_performance_analysis(metrics, output_dir)
        
        print("Performance visualizations saved to: {}".format(output_dir))
    
    def _create_roi_comparison_chart(self, metrics: Dict[str, Any], output_dir: str):
        """Create ROI comparison chart"""
        individual_metrics = metrics['individual_metrics']
        
        agents = list(individual_metrics.keys())
        rois = [individual_metrics[agent]['roi'] for agent in agents]
        
        fig = go.Figure(data=[
            go.Bar(x=agents, y=rois, 
                  marker_color=['green' if roi > 0 else 'red' for roi in rois])
        ])
        
        fig.update_layout(
            title='Agent ROI Comparison',
            xaxis_title='Agent',
            yaxis_title='ROI (%)',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'roi_comparison.html'))
        fig.write_image(os.path.join(output_dir, 'roi_comparison.png'))
    
    def _create_risk_return_scatter(self, metrics: Dict[str, Any], output_dir: str):
        """Create risk-return scatter plot"""
        individual_metrics = metrics['individual_metrics']
        risk_metrics = metrics['risk_metrics']
        
        agents = list(individual_metrics.keys())
        returns = [individual_metrics[agent]['roi'] for agent in agents]
        risks = [risk_metrics[agent]['total_risk_score'] for agent in agents]
        
        fig = go.Figure(data=go.Scatter(
            x=risks, y=returns, mode='markers+text',
            text=agents, textposition="top center",
            marker=dict(size=10, color=returns, colorscale='RdYlGn', showscale=True)
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Risk Score',
            yaxis_title='Return (%)',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'risk_return_scatter.html'))
        fig.write_image(os.path.join(output_dir, 'risk_return_scatter.png'))
    
    def _create_efficiency_analysis(self, metrics: Dict[str, Any], output_dir: str):
        """Create efficiency analysis chart"""
        efficiency_metrics = metrics['efficiency_metrics']
        
        agents = list(efficiency_metrics.keys())
        gas_eff = [efficiency_metrics[agent]['gas_efficiency'] for agent in agents]
        txn_eff = [efficiency_metrics[agent]['transaction_efficiency'] for agent in agents]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Gas Efficiency', x=agents, y=gas_eff))
        fig.add_trace(go.Bar(name='Transaction Efficiency', x=agents, y=txn_eff))
        
        fig.update_layout(
            title='Agent Efficiency Comparison',
            xaxis_title='Agent',
            yaxis_title='Efficiency Score',
            barmode='group',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'efficiency_analysis.html'))
        fig.write_image(os.path.join(output_dir, 'efficiency_analysis.png'))
    
    def _create_gas_optimization_chart(self, metrics: Dict[str, Any], output_dir: str):
        """Create gas optimization analysis chart"""
        individual_metrics = metrics['individual_metrics']
        
        # Filter agents with gas optimization data
        gas_agents = {agent: data for agent, data in individual_metrics.items() 
                     if 'gas_optimization' in data}
        
        if not gas_agents:
            return
        
        agents = list(gas_agents.keys())
        delays = [gas_agents[agent]['gas_optimization']['total_delays'] for agent in agents]
        savings = [gas_agents[agent]['gas_optimization']['total_gas_saved'] for agent in agents]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=agents, y=delays, name="Total Delays"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=agents, y=savings, mode='lines+markers', name="Gas Saved"),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Agent")
        fig.update_yaxes(title_text="Number of Delays", secondary_y=False)
        fig.update_yaxes(title_text="Gas Saved", secondary_y=True)
        
        fig.update_layout(title_text="Gas Optimization Performance")
        
        fig.write_html(os.path.join(output_dir, 'gas_optimization.html'))
        fig.write_image(os.path.join(output_dir, 'gas_optimization.png'))
    
    def _create_market_performance_analysis(self, metrics: Dict[str, Any], output_dir: str):
        """Create market performance analysis"""
        market_adjusted = metrics['market_adjusted_metrics']
        
        agents = list(market_adjusted.keys())
        alphas = [market_adjusted[agent]['alpha'] for agent in agents]
        betas = [market_adjusted[agent]['beta'] for agent in agents]
        
        fig = go.Figure(data=go.Scatter(
            x=betas, y=alphas, mode='markers+text',
            text=agents, textposition="top center",
            marker=dict(size=12, color=alphas, colorscale='RdYlGn', showscale=True)
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=1, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Alpha vs Beta Analysis',
            xaxis_title='Beta (Market Sensitivity)',
            yaxis_title='Alpha (Excess Return)',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'market_performance.html'))
        fig.write_image(os.path.join(output_dir, 'market_performance.png'))

def main():
    """Demo of performance evaluator"""
    # This would typically load results from a file
    print("Performance Evaluator Demo")
    print("To use: evaluator = PerformanceEvaluator()")
    print("        evaluator.load_results('experiment_results.json')")
    print("        evaluator.generate_performance_report()")
    print("        evaluator.create_performance_visualizations()")

if __name__ == "__main__":
    main() 