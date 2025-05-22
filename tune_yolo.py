#!/usr/bin/env python3
"""
YOLOv12 Hyperparameter Tuning for Powerline Detection

This script performs hyperparameter tuning for YOLOv12 models with a focus on maximizing recall 
for powerline detection. Since this model will be used as a filtering step, we prioritize 
minimizing false negatives over precision.

Usage:
    python tune_yolo.py

Requirements:
    - ultralytics
    - PyTorch
    - pandas
    - matplotlib
    - seaborn
    - PyYAML
"""

import yaml
import os
import sys
import platform
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO, settings
from ultralytics.data.utils import check_det_dataset
import json
from datetime import datetime
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tune_yolo.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOHyperparameterTuner:
    """
    A class to handle YOLO model hyperparameter tuning with focus on powerline detection.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the tuner with configuration.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path().resolve()
        self.tune_config = self.config['tune']
        self.results_dir = None

        # Setup
        self._setup_environment()
        self._verify_dataset()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise

    def _setup_environment(self) -> None:
        """Setup the environment and display system information."""
        # Configure Ultralytics datasets directory
        settings.update({"datasets_dir": str(self.project_root)})
        logger.info(
            f"✅ Ultralytics datasets_dir set to: {settings['datasets_dir']}")

        # Display system information
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Dataset YAML: {self.config['dataset_yaml_path']}")
        logger.info(f"Model type: {self.config['model_type']}")
        logger.info(f"Device: {self.config['device']}")

    def _verify_dataset(self) -> None:
        """Verify and clean dataset configuration."""
        dataset_yaml_path = Path(self.config["dataset_yaml_path"]).resolve()

        if not dataset_yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {dataset_yaml_path}")

        # Clean up dataset YAML if needed
        try:
            data = yaml.safe_load(dataset_yaml_path.read_text())
            modified = False

            if data.pop("path", None) is not None:
                logger.info("🔄 Removed stale 'path' key from dataset YAML")
                modified = True

            # Ensure single class for powerline detection
            if data.get("nc") != 1 or data.get("names") != ["powerline"]:
                data["nc"] = 1
                data["names"] = ["powerline"]
                modified = True
                logger.info("🔄 Updated dataset to single class 'powerline'")

            if modified:
                dataset_yaml_path.write_text(
                    yaml.safe_dump(data, sort_keys=False))
                logger.info(f"✅ Dataset YAML cleaned: {dataset_yaml_path}")

            # Verify dataset structure
            check_det_dataset(str(dataset_yaml_path))
            logger.info("✅ Dataset structure verified")

        except Exception as e:
            logger.warning(f"⚠️ Dataset verification warning: {e}")

    def _prepare_search_space(self) -> Dict[str, Tuple[float, float]]:
        """Convert search space from config format to ultralytics format."""
        search_space = {}

        for param, value_range in self.tune_config['search_space'].items():
            if isinstance(value_range, list) and len(value_range) == 2:
                search_space[param] = tuple(value_range)
            else:
                logger.warning(
                    f"⚠️ Skipping parameter {param}: invalid range format {value_range}")

        logger.info(
            f"Search space prepared with {len(search_space)} parameters:")
        for param, range_vals in search_space.items():
            logger.info(f"  {param}: {range_vals}")

        return search_space

    def _initialize_model(self) -> YOLO:
        """Initialize the YOLO model."""
        model_path = self.config['model_type']
        logger.info(f"🚀 Initializing model: {model_path}")

        try:
            model = YOLO(model_path)
            logger.info(f"✅ Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def run_tuning(self) -> Optional[Any]:
        """
        Execute the hyperparameter tuning process.

        Returns:
            Tuning results object or None if failed
        """
        # Prepare search space and model
        search_space = self._prepare_search_space()
        model = self._initialize_model()

        # Prepare dataset path
        dataset_yaml_path = Path(self.config["dataset_yaml_path"]).resolve()

        # Prepare tuning arguments
        tuning_args = {
            'data': str(dataset_yaml_path),
            'epochs': self.tune_config['epochs'],
            'iterations': self.tune_config['iterations'],
            'optimizer': self.tune_config['optimizer'],
            'plots': self.tune_config['plots'],
            'save': self.tune_config['save'],
            'val': self.tune_config['val'],
            'device': self.config['device'],
            'patience': self.tune_config['patience'],
            'space': search_space,
            'project': self.config['output_config']['project'] + '_tune',
            'name': f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True
        }

        # Add resume capability
        if self.tune_config.get('resume', False):
            tuning_args['resume'] = True

        # Log tuning configuration
        logger.info("Tuning configuration:")
        for key, value in tuning_args.items():
            if key != 'space':  # Don't print the entire search space again
                logger.info(f"  {key}: {value}")

        # Calculate estimated time
        estimated_time_min = self.tune_config['epochs'] * \
            self.tune_config['iterations'] / 60
        logger.info(f"🎯 Starting hyperparameter tuning...")
        logger.info(f"⏱️ Estimated time: {estimated_time_min:.1f} minutes")
        logger.info(
            f"🎯 Focus: Maximizing recall for powerline detection filtering")

        # Start tuning
        start_time = datetime.now()
        logger.info(
            f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            results = model.tune(**tuning_args)

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info(f"✅ Hyperparameter tuning completed!")
            logger.info(
                f"🕐 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ Total duration: {duration}")

            if hasattr(results, 'save_dir'):
                self.results_dir = Path(results.save_dir)
                logger.info(f"📁 Results saved to: {self.results_dir}")
            else:
                logger.warning(
                    "⚠️ Results directory not found in results object")

            return results

        except Exception as e:
            logger.error(f"❌ Error during tuning: {e}")
            return None

    def _find_tune_results(self, project_dir: str, run_name: str = None) -> Optional[Path]:
        """Find the tune results directory."""
        try:
            project_path = Path(project_dir)

            if run_name:
                tune_dir = project_path / run_name / 'tune'
            else:
                # Find the most recent tune directory
                tune_dirs = list(project_path.glob('*/tune'))
                if not tune_dirs:
                    raise FileNotFoundError(
                        f"No tune directories found in {project_path}")
                tune_dir = max(tune_dirs, key=lambda x: x.stat().st_mtime)

            if not tune_dir.exists():
                raise FileNotFoundError(
                    f"Tune directory not found: {tune_dir}")

            return tune_dir
        except Exception as e:
            logger.error(f"⚠️ Could not find tune results: {e}")
            return None

    def analyze_results(self) -> Optional[pd.DataFrame]:
        """
        Analyze tuning results and create visualizations.

        Returns:
            DataFrame with results or None if not available
        """
        # Find results directory
        if not self.results_dir:
            project_dir = self.config['output_config']['project'] + '_tune'
            self.results_dir = self._find_tune_results(project_dir)

        if not self.results_dir:
            logger.error("⚠️ No results directory found for analysis")
            return None

        logger.info(f"📊 Analyzing results from: {self.results_dir}")

        # Check available files
        available_files = list(self.results_dir.glob('*'))
        logger.info("Available result files:")
        for file_path in available_files:
            logger.info(f"  {file_path.name}")

        # Load and analyze tune results CSV
        results_csv_path = self.results_dir / 'tune_results.csv'
        if not results_csv_path.exists():
            logger.warning(
                "⚠️ tune_results.csv not found - cannot analyze detailed results")
            return None

        try:
            results_df = pd.read_csv(results_csv_path)
            logger.info(f"📈 Loaded {len(results_df)} tuning iterations")

            # Display basic statistics
            logger.info("📊 Results Summary:")
            logger.info(
                f"Best fitness score: {results_df['fitness'].max():.4f}")
            logger.info(
                f"Average fitness score: {results_df['fitness'].mean():.4f}")
            logger.info(
                f"Std fitness score: {results_df['fitness'].std():.4f}")

            # Find best iteration
            best_idx = results_df['fitness'].idxmax()
            best_result = results_df.iloc[best_idx]

            logger.info(f"🏆 Best iteration (#{best_idx}):")
            logger.info(f"  Fitness: {best_result['fitness']:.4f}")

            # Display best hyperparameters
            hyperparam_cols = [
                col for col in results_df.columns if col != 'fitness']
            logger.info("🎛️ Best hyperparameters:")
            # Show first 15 to avoid cluttering
            for col in hyperparam_cols[:15]:
                if col in best_result:
                    logger.info(f"  {col}: {best_result[col]}")

            if len(hyperparam_cols) > 15:
                logger.info(
                    f"  ... and {len(hyperparam_cols) - 15} more parameters")

            # Create visualizations
            self._create_visualizations(results_df)

            return results_df

        except Exception as e:
            logger.error(f"❌ Error analyzing results: {e}")
            return None

    def _create_visualizations(self, results_df: pd.DataFrame) -> None:
        """Create and save visualization plots."""
        try:
            # Set style for plots
            plt.style.use('default')
            sns.set_palette("husl")

            # Set up the plotting area
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hyperparameter Tuning Results Analysis',
                         fontsize=16, fontweight='bold')

            # 1. Fitness score progression
            axes[0, 0].plot(results_df.index,
                            results_df['fitness'], alpha=0.7, linewidth=1)
            axes[0, 0].scatter(
                results_df.index, results_df['fitness'], alpha=0.5, s=20)
            axes[0, 0].axhline(y=results_df['fitness'].max(
            ), color='red', linestyle='--', alpha=0.7, label='Best')
            axes[0, 0].set_title('Fitness Score Progression')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Fitness Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Fitness distribution
            axes[0, 1].hist(results_df['fitness'], bins=20,
                            alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=results_df['fitness'].mean(
            ), color='red', linestyle='--', alpha=0.7, label='Mean')
            axes[0, 1].axvline(x=results_df['fitness'].max(
            ), color='green', linestyle='--', alpha=0.7, label='Best')
            axes[0, 1].set_title('Fitness Score Distribution')
            axes[0, 1].set_xlabel('Fitness Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Top 10 iterations comparison
            top_10 = results_df.nlargest(10, 'fitness')
            axes[1, 0].bar(range(len(top_10)), top_10['fitness'])
            axes[1, 0].set_title('Top 10 Fitness Scores')
            axes[1, 0].set_xlabel('Rank')
            axes[1, 0].set_ylabel('Fitness Score')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Parameter impact analysis
            if len(results_df) > 20:
                # Find parameters most correlated with fitness
                numeric_cols = results_df.select_dtypes(
                    include=[np.number]).columns
                correlations = results_df[numeric_cols].corrwith(
                    results_df['fitness']).abs().sort_values(ascending=False)
                # Exclude fitness itself, take top 7
                top_correlations = correlations[1:8]

                if len(top_correlations) > 0:
                    axes[1, 1].barh(range(len(top_correlations)),
                                    top_correlations.values)
                    axes[1, 1].set_yticks(range(len(top_correlations)))
                    axes[1, 1].set_yticklabels(
                        top_correlations.index, fontsize=8)
                    axes[1, 1].set_title('Parameter Impact on Fitness')
                    axes[1, 1].set_xlabel('Correlation with Fitness')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No correlation data available',
                                    ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Parameter Impact (N/A)')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for analysis\n(need >20 iterations)',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Parameter Impact (N/A)')

            plt.tight_layout()

            # Save the plot
            if self.results_dir:
                plot_path = self.results_dir / 'tuning_analysis.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Analysis plot saved to: {plot_path}")

            plt.close(fig)  # Close to free memory

        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")

    def extract_best_hyperparameters(self) -> None:
        """Extract and save the best hyperparameters."""
        if not self.results_dir:
            logger.warning(
                "⚠️ No results directory available for hyperparameter extraction")
            return

        best_hyperparams_path = self.results_dir / 'best_hyperparameters.yaml'

        if not best_hyperparams_path.exists():
            logger.warning("⚠️ best_hyperparameters.yaml not found")
            available_files = list(self.results_dir.glob('*'))
            logger.info(f"Available files in {self.results_dir}:")
            for file_path in available_files:
                logger.info(f"  {file_path.name}")
            return

        try:
            with open(best_hyperparams_path, 'r') as f:
                best_hyperparams_content = f.read()

            logger.info(
                "🏆 Best Hyperparameters (from best_hyperparameters.yaml):")
            logger.info("=" * 60)
            print(best_hyperparams_content)  # Use print for better formatting

            # Try to parse as YAML for further processing
            try:
                # Extract the hyperparameters section
                if '# Best fitness hyperparameters are printed below.' in best_hyperparams_content:
                    hyperparams_section = best_hyperparams_content.split(
                        '# Best fitness hyperparameters are printed below.')[-1]
                    best_hyperparams = yaml.safe_load(hyperparams_section)

                    if best_hyperparams:
                        # Create a formatted version for easy copying
                        formatted_config = {
                            'hyperparams': best_hyperparams,
                            'tuning_info': {
                                'tuned_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'iterations': self.tune_config['iterations'],
                                'epochs_per_iteration': self.tune_config['epochs'],
                                'focus': 'Powerline detection with high recall'
                            }
                        }

                        # Save optimized config
                        optimized_config_path = self.results_dir / 'optimized_config.yaml'
                        with open(optimized_config_path, 'w') as f:
                            yaml.dump(formatted_config, f,
                                      default_flow_style=False, sort_keys=False)

                        logger.info(
                            f"💾 Optimized configuration saved to: {optimized_config_path}")

            except Exception as e:
                logger.warning(
                    f"⚠️ Could not parse hyperparameters for formatting: {e}")

            logger.info(
                "📋 You can now update your main config.yaml with these hyperparameters for optimal powerline detection!")

        except Exception as e:
            logger.error(f"❌ Error extracting hyperparameters: {e}")

    def print_summary(self, results_df: Optional[pd.DataFrame] = None) -> None:
        """Print a summary of the tuning process."""
        logger.info("🎯 Hyperparameter Tuning Summary")
        logger.info("=" * 50)

        if results_df is not None:
            logger.info(f"✅ Completed {len(results_df)} tuning iterations")
            logger.info(
                f"🏆 Best fitness score: {results_df['fitness'].max():.4f}")
            if len(results_df) > 1:
                improvement = (
                    (results_df['fitness'].max() / results_df['fitness'].iloc[0]) - 1) * 100
                logger.info(
                    f"📈 Improvement over first iteration: {improvement:.1f}%")
        else:
            logger.warning("⚠️ Results not available for summary")

        if self.results_dir:
            logger.info(f"📁 Results location: {self.results_dir}")

        logger.info("")
        logger.info("🔄 Next Steps:")
        logger.info("1. Review the best hyperparameters above")
        logger.info("2. Update your main config.yaml with the optimized values")
        logger.info(
            "3. Run a full training session with the optimized hyperparameters")
        logger.info(
            "4. Evaluate the model performance on your validation/test set")
        logger.info("5. Consider running additional tuning if needed")

        logger.info("")
        logger.info("💡 Tips for Powerline Detection:")
        logger.info("- Focus on recall metrics since this is a filtering step")
        logger.info(
            "- Consider the trade-off between recall and inference speed")
        logger.info(
            "- Test the model on diverse lighting and weather conditions")
        logger.info(
            "- Monitor for overfitting to your specific powerline types")

        logger.info("")
        logger.info(
            "✨ Happy training! Your model should now be optimized for powerline detection.")


def main():
    """Main function to run the hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description='YOLO Hyperparameter Tuning for Powerline Detection')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze existing results, skip tuning')

    args = parser.parse_args()

    try:
        # Initialize tuner
        tuner = YOLOHyperparameterTuner(args.config)

        results_df = None

        if not args.analyze_only:
            # Run tuning
            logger.info("🚀 Starting hyperparameter tuning process...")
            results = tuner.run_tuning()

            if results is None:
                logger.error(
                    "❌ Tuning failed, but will attempt to analyze any existing results")

        # Analyze results
        logger.info("📊 Analyzing results...")
        results_df = tuner.analyze_results()

        # Extract best hyperparameters
        logger.info("🏆 Extracting best hyperparameters...")
        tuner.extract_best_hyperparameters()

        # Print summary
        tuner.print_summary(results_df)

        logger.info("🎉 Hyperparameter tuning process completed!")

    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
