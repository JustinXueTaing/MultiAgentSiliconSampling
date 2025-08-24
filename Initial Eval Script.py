import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SiliconSamplingEvaluator:
    """
    Evaluates how well synthetic silicon sampling data matches reference survey data
    across demographic groups and response distributions.
    """
    
    def __init__(self, reference_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Initialize evaluator with reference and synthetic datasets.
        
        Args:
            reference_data: DataFrame with real survey responses and demographics
            synthetic_data: DataFrame with LLM-generated responses and demographics
        """
        self.reference_data = reference_data
        self.synthetic_data = synthetic_data
        self.results = {}
        
    def preprocess_data(self):
        """Clean and standardize both datasets for comparison."""
        # Ensure common columns exist
        common_cols = set(self.reference_data.columns) & set(self.synthetic_data.columns)
        
        if len(common_cols) < 2:
            raise ValueError("Reference and synthetic data must share at least 2 columns")
            
        # Standardize categorical variables
        for col in common_cols:
            if self.reference_data[col].dtype == 'object':
                # Get union of all categories
                all_cats = set(self.reference_data[col].unique()) | set(self.synthetic_data[col].unique())
                
                # Convert to categorical with same categories
                self.reference_data[col] = pd.Categorical(self.reference_data[col], categories=all_cats)
                self.synthetic_data[col] = pd.Categorical(self.synthetic_data[col], categories=all_cats)
    
    def compute_distributional_fit(self, target_column: str, 
                                 demographic_columns: List[str] = None) -> Dict[str, float]:
        """
        Compute distributional fit metrics between reference and synthetic data.
        
        Args:
            target_column: Column containing survey responses to compare
            demographic_columns: List of demographic columns for stratified analysis
            
        Returns:
            Dictionary of distributional fit metrics
        """
        metrics = {}
        
        # Overall distribution comparison
        ref_dist = self.reference_data[target_column].value_counts(normalize=True).sort_index()
        syn_dist = self.synthetic_data[target_column].value_counts(normalize=True).sort_index()
        
        # Align distributions (handle missing categories)
        all_categories = ref_dist.index.union(syn_dist.index)
        ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
        syn_aligned = syn_dist.reindex(all_categories, fill_value=0)
        
        # Jensen-Shannon Divergence
        js_div = jensenshannon(ref_aligned.values, syn_aligned.values)
        metrics['jensen_shannon_divergence'] = js_div
        
        # KL Divergence (with smoothing)
        ref_smooth = ref_aligned.values + 1e-10
        syn_smooth = syn_aligned.values + 1e-10
        ref_smooth /= ref_smooth.sum()
        syn_smooth /= syn_smooth.sum()
        
        kl_div = stats.entropy(syn_smooth, ref_smooth)
        metrics['kl_divergence'] = kl_div
        
        # Chi-square test
        try:
            ref_counts = self.reference_data[target_column].value_counts().sort_index()
            syn_counts = self.synthetic_data[target_column].value_counts().sort_index()
            
            # Align counts
            ref_counts_aligned = ref_counts.reindex(all_categories, fill_value=0)
            syn_counts_aligned = syn_counts.reindex(all_categories, fill_value=0)
            
            contingency_table = np.array([ref_counts_aligned.values, syn_counts_aligned.values])
            chi2, p_val = chi2_contingency(contingency_table)[:2]
            
            metrics['chi2_statistic'] = chi2
            metrics['chi2_p_value'] = p_val
        except:
            metrics['chi2_statistic'] = np.nan
            metrics['chi2_p_value'] = np.nan
        
        # Demographic stratified analysis
        if demographic_columns:
            stratified_metrics = {}
            
            for demo_col in demographic_columns:
                demo_groups = set(self.reference_data[demo_col].unique()) | \
                             set(self.synthetic_data[demo_col].unique())
                
                group_js_divs = []
                
                for group in demo_groups:
                    ref_group = self.reference_data[self.reference_data[demo_col] == group]
                    syn_group = self.synthetic_data[self.synthetic_data[demo_col] == group]
                    
                    if len(ref_group) > 0 and len(syn_group) > 0:
                        ref_group_dist = ref_group[target_column].value_counts(normalize=True).sort_index()
                        syn_group_dist = syn_group[target_column].value_counts(normalize=True).sort_index()
                        
                        # Align distributions
                        group_categories = ref_group_dist.index.union(syn_group_dist.index)
                        ref_group_aligned = ref_group_dist.reindex(group_categories, fill_value=0)
                        syn_group_aligned = syn_group_dist.reindex(group_categories, fill_value=0)
                        
                        group_js = jensenshannon(ref_group_aligned.values, syn_group_aligned.values)
                        group_js_divs.append(group_js)
                
                stratified_metrics[f'{demo_col}_mean_js_divergence'] = np.mean(group_js_divs) if group_js_divs else np.nan
                stratified_metrics[f'{demo_col}_std_js_divergence'] = np.std(group_js_divs) if group_js_divs else np.nan
            
            metrics.update(stratified_metrics)
        
        return metrics
    
    def compute_demographic_fidelity(self, demographic_columns: List[str]) -> Dict[str, float]:
        """
        Assess how well synthetic data preserves demographic distributions.
        
        Args:
            demographic_columns: List of demographic columns to analyze
            
        Returns:
            Dictionary of demographic fidelity metrics
        """
        metrics = {}
        
        for demo_col in demographic_columns:
            ref_demo_dist = self.reference_data[demo_col].value_counts(normalize=True).sort_index()
            syn_demo_dist = self.synthetic_data[demo_col].value_counts(normalize=True).sort_index()
            
            # Align distributions
            all_demo_cats = ref_demo_dist.index.union(syn_demo_dist.index)
            ref_demo_aligned = ref_demo_dist.reindex(all_demo_cats, fill_value=0)
            syn_demo_aligned = syn_demo_dist.reindex(all_demo_cats, fill_value=0)
            
            # Jensen-Shannon divergence for demographic distribution
            demo_js = jensenshannon(ref_demo_aligned.values, syn_demo_aligned.values)
            metrics[f'{demo_col}_demographic_js_divergence'] = demo_js
            
        return metrics
    
    def bootstrap_confidence_intervals(self, target_column: str, 
                                     metric_func, n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for distributional metrics.
        
        Args:
            target_column: Column to analyze
            metric_func: Function that computes the metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_metrics = []
        
        ref_size = len(self.reference_data)
        syn_size = len(self.synthetic_data)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample both datasets
            ref_bootstrap = self.reference_data.sample(n=ref_size, replace=True)
            syn_bootstrap = self.synthetic_data.sample(n=syn_size, replace=True)
            
            # Create temporary evaluator
            temp_evaluator = SiliconSamplingEvaluator(ref_bootstrap, syn_bootstrap)
            
            # Compute metric
            metric_val = metric_func(temp_evaluator, target_column)
            bootstrap_metrics.append(metric_val)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        
        return lower_bound, upper_bound
    
    def _interpret_js_divergence(self, js_value: float) -> str:
        """Simple interpretation of JS Divergence values."""
        if js_value < 0.1:
            return "EXCELLENT"
        elif js_value < 0.3:
            return "GOOD"
        elif js_value < 0.5:
            return "FAIR"
        else:
            return "POOR"
    
    def generate_comparison_report(self, target_column: str, 
                                 demographic_columns: List[str] = None,
                                 save_path: str = None) -> Dict[str, Any]:
        """
        Generate simplified comparison report focused on JS Divergence.
        
        Args:
            target_column: Main response column to analyze
            demographic_columns: Demographic columns for stratified analysis
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary containing evaluation results
        """
        print("Silicon Sampling Evaluation Report")
        print("=" * 40)
        
        # Preprocess data
        self.preprocess_data()
        
        # Compute distributional fit
        dist_metrics = self.compute_distributional_fit(target_column, demographic_columns)
        
        # Compute demographic fidelity
        demo_metrics = {}
        if demographic_columns:
            demo_metrics = self.compute_demographic_fidelity(demographic_columns)
        
        # Get main JS divergence
        main_js = dist_metrics.get('jensen_shannon_divergence', float('nan'))
        main_interpretation = self._interpret_js_divergence(main_js)
        
        # Print simplified results
        print(f"\nOverall Match for '{target_column}':")
        print(f"JS Divergence: {main_js:.3f} ({main_interpretation})")
        
        if demographic_columns:
            print(f"\nDemographic Group Matches:")
            for demo_col in demographic_columns:
                demo_js = demo_metrics.get(f'{demo_col}_demographic_js_divergence', float('nan'))
                demo_interpretation = self._interpret_js_divergence(demo_js)
                print(f"{demo_col}: {demo_js:.3f} ({demo_interpretation})")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"Your synthetic data has {main_interpretation.lower()} overall similarity to reference data.")
        
        # Generate visualizations
        if save_path:
            self._create_visualizations(target_column, demographic_columns, save_path)
        
        return {
            'overall_js_divergence': main_js,
            'overall_quality': main_interpretation,
            'demographic_results': {col: demo_metrics.get(f'{col}_demographic_js_divergence', float('nan')) 
                                  for col in (demographic_columns or [])},
            'full_metrics': {**dist_metrics, **demo_metrics}
        }
    
    def _create_visualizations(self, target_column: str, 
                              demographic_columns: List[str], save_path: str):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Silicon Sampling vs Reference Data Comparison', fontsize=16)
        
        # Overall response distribution comparison
        ref_dist = self.reference_data[target_column].value_counts(normalize=True).sort_index()
        syn_dist = self.synthetic_data[target_column].value_counts(normalize=True).sort_index()
        
        # Align for plotting
        all_cats = ref_dist.index.union(syn_dist.index)
        ref_aligned = ref_dist.reindex(all_cats, fill_value=0)
        syn_aligned = syn_dist.reindex(all_cats, fill_value=0)
        
        x = np.arange(len(all_cats))
        width = 0.35
        
        axes[0,0].bar(x - width/2, ref_aligned.values, width, label='Reference', alpha=0.7)
        axes[0,0].bar(x + width/2, syn_aligned.values, width, label='Synthetic', alpha=0.7)
        axes[0,0].set_title(f'{target_column} Distribution Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(all_cats, rotation=45)
        axes[0,0].legend()
        
        # Demographic distribution comparison (if available)
        if demographic_columns and len(demographic_columns) > 0:
            demo_col = demographic_columns[0]
            ref_demo = self.reference_data[demo_col].value_counts(normalize=True).sort_index()
            syn_demo = self.synthetic_data[demo_col].value_counts(normalize=True).sort_index()
            
            demo_cats = ref_demo.index.union(syn_demo.index)
            ref_demo_aligned = ref_demo.reindex(demo_cats, fill_value=0)
            syn_demo_aligned = syn_demo.reindex(demo_cats, fill_value=0)
            
            x_demo = np.arange(len(demo_cats))
            axes[0,1].bar(x_demo - width/2, ref_demo_aligned.values, width, 
                         label='Reference', alpha=0.7)
            axes[0,1].bar(x_demo + width/2, syn_demo_aligned.values, width, 
                         label='Synthetic', alpha=0.7)
            axes[0,1].set_title(f'{demo_col} Distribution Comparison')
            axes[0,1].set_xticks(x_demo)
            axes[0,1].set_xticklabels(demo_cats, rotation=45)
            axes[0,1].legend()
        
        # Q-Q plot for numerical comparison
        if pd.api.types.is_numeric_dtype(self.reference_data[target_column]):
            ref_numeric = pd.to_numeric(self.reference_data[target_column], errors='coerce').dropna()
            syn_numeric = pd.to_numeric(self.synthetic_data[target_column], errors='coerce').dropna()
            
            stats.probplot(ref_numeric, dist="norm", plot=axes[1,0])
            axes[1,0].set_title('Reference Data Q-Q Plot')
            
            stats.probplot(syn_numeric, dist="norm", plot=axes[1,1])
            axes[1,1].set_title('Synthetic Data Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

# Example usage function
def example_usage():
    """
    Example of how to use the SiliconSamplingEvaluator.
    Replace with your actual data loading.
    """
    # Create example data (replace with your actual data loading)
    np.random.seed(42)
    
    # Example reference data (simulating ANES/CES/GSS structure)
    n_ref = 5000
    reference_data = pd.DataFrame({
        'age': np.random.choice(['18-29', '30-44', '45-64', '65+'], n_ref, 
                               p=[0.2, 0.25, 0.35, 0.2]),
        'education': np.random.choice(['High School', 'College', 'Graduate'], n_ref,
                                    p=[0.3, 0.5, 0.2]),
        'ideology': np.random.choice(['Liberal', 'Moderate', 'Conservative'], n_ref,
                                   p=[0.3, 0.4, 0.3]),
        'response': np.random.choice(['Strongly Agree', 'Agree', 'Disagree', 'Strongly Disagree'], 
                                   n_ref, p=[0.15, 0.35, 0.35, 0.15])
    })
    
    # Example synthetic data (slightly different distributions to show differences)
    n_syn = 3000
    synthetic_data = pd.DataFrame({
        'age': np.random.choice(['18-29', '30-44', '45-64', '65+'], n_syn,
                               p=[0.25, 0.3, 0.3, 0.15]),  # Slightly different
        'education': np.random.choice(['High School', 'College', 'Graduate'], n_syn,
                                    p=[0.25, 0.55, 0.2]),    # Slightly different
        'ideology': np.random.choice(['Liberal', 'Moderate', 'Conservative'], n_syn,
                                   p=[0.35, 0.35, 0.3]),    # Slightly different
        'response': np.random.choice(['Strongly Agree', 'Agree', 'Disagree', 'Strongly Disagree'], 
                                   n_syn, p=[0.2, 0.4, 0.3, 0.1])  # Different response pattern
    })
    
    # Initialize evaluator
    evaluator = SiliconSamplingEvaluator(reference_data, synthetic_data)
    
    # Generate report
    results = evaluator.generate_comparison_report(
        target_column='response',
        demographic_columns=['age', 'education', 'ideology'],
        save_path='silicon_sampling_eval'
    )
    
    return results

if __name__ == "__main__":
    results = example_usage()
    print("\nEvaluation complete! Check the results dictionary for detailed metrics.")