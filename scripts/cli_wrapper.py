#!/usr/bin/env python3
"""
Command-line interface for Data Science Project Setup
Usage: python setup_project.py --slug my-project --dataset owner/dataset-name
"""

import argparse
import sys
from pathlib import Path

# Import the main setup class (assuming it's in the same directory)
from ds_project_setup import DataScienceProjectSetup


def main():
    parser = argparse.ArgumentParser(
        description="Create a standardized data science project structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_project.py --slug house-prices --dataset house-prices-advanced-regression-techniques
  python setup_project.py --slug customer-churn --dataset blastchar/telco-customer-churn --root /path/to/project
        """
    )
    
    parser.add_argument(
        '--slug', '-s',
        required=True,
        help='Project identifier (used for folder names)'
    )
    
    parser.add_argument(
        '--dataset', '-d', 
        required=True,
        help='Kaggle dataset key (format: owner/dataset-name)'
    )
    
    parser.add_argument(
        '--root', '-r',
        default=None,
        help='Git root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating anything'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.slug.replace('-', '').replace('_', '').isalnum():
        print("❌ Error: SLUG must contain only alphanumeric characters, hyphens, and underscores")
        sys.exit(1)
    
    if '/' not in args.dataset:
        print("❌ Error: Dataset key must be in format 'owner/dataset-name'")
        sys.exit(1)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be created")
        print(f"Would create project: {args.slug}")
        print(f"Would download dataset: {args.dataset}")
        print(f"Git root: {args.root or 'current directory'}")
        return
    
    try:
        # Create and run setup
        setup = DataScienceProjectSetup(
            slug=args.slug,
            dataset_key=args.dataset,
            git_root=args.root
        )
        
        setup.run_setup()
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()