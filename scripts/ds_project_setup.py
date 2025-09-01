#!/usr/bin/env python3
"""
Data Science Project Setup Script
Creates standardized project structure and setup notebook for EDA projects.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
import kagglehub as kagglehub

class DataScienceProjectSetup:
    """Automates creation of standardized DS project structure."""
    
    def __init__(self, slug: str, dataset_key: str, git_root: Optional[str] = None):
        self.slug = slug
        self.dataset_key = dataset_key
        self.git_root = Path(git_root) if git_root else Path.cwd()
        
        # Define directory structure
        self.base_folders = ['data', 'notebooks', 'reports', 'figures']
        self.slug_dirs = {
            'data': self.git_root / 'data' / slug,
            'notebooks': self.git_root / 'notebooks' / slug,
            'reports': self.git_root / 'reports' / slug,
            'figures': self.git_root / 'figures' / slug
        }
    
    def create_directory_structure(self) -> None:
        """Create the required directory structure."""
        print(f"Creating directory structure for project: {self.slug}")
        
        for folder_name in self.base_folders:
            slug_dir = self.slug_dirs[folder_name]
            slug_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {slug_dir}")
    
    def generate_setup_notebook(self) -> None:
        """Generate the setup.ipynb notebook with all required functionality."""
        notebook_path = self.slug_dirs['notebooks'] / 'setup.ipynb'
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# Project Setup: {self.slug}\n",
                        "This notebook initializes the analysis environment and downloads required data.\n",
                        "Run all cells to set up your analysis environment."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Essential imports\n",
                        "import os\n",
                        "import sys\n",
                        "import shutil\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "from pathlib import Path\n",
                        "import kagglehub\n",
                        "from IPython.display import display\n",
                        "\n",
                        "print(\"✓ Imports completed successfully\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Project Configuration - These variables are shared across all notebooks\n",
                        f"SLUG = '{self.slug}'\n",
                        f"DATASET_KEY = '{self.dataset_key}'\n",
                        "\n",
                        "# Directory paths\n",
                        f"GIT_ROOT = Path('{self.git_root}')\n",
                        "DATA_DIR = GIT_ROOT / 'data' / SLUG\n",
                        "FIG_DIR = GIT_ROOT / 'figures' / SLUG\n",
                        "REP_DIR = GIT_ROOT / 'reports' / SLUG\n",
                        "NOTEBOOK_DIR = GIT_ROOT / 'notebooks' / SLUG\n",
                        "\n",
                        "# Make variables available to other notebooks in this folder\n",
                        "%store SLUG\n",
                        "%store DATA_DIR\n",
                        "%store FIG_DIR\n",
                        "%store REP_DIR\n",
                        "%store NOTEBOOK_DIR\n",
                        "%store DATASET_KEY\n",
                        "\n",
                        "print(f\"Project: {SLUG}\")\n",
                        "print(f\"Data Directory: {DATA_DIR}\")\n",
                        "print(f\"Figures Directory: {FIG_DIR}\")\n",
                        "print(f\"Reports Directory: {REP_DIR}\")\n",
                        "print(\"\\n✓ Configuration variables set and stored\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Download dataset from Kaggle\n",
                        "print(f\"Downloading dataset: {DATASET_KEY}\")\n",
                        "print(\"This may take a few minutes depending on dataset size...\")\n",
                        "\n",
                        "try:\n",
                        "    download_path = kagglehub.dataset_download(DATASET_KEY)\n",
                        "    print(f\"✓ Dataset downloaded successfully to: {download_path}\")\n",
                        "    DOWNLOAD_PATH = Path(download_path)\n",
                        "    %store DOWNLOAD_PATH\n",
                        "except Exception as e:\n",
                        "    print(f\"❌ Error downloading dataset: {e}\")\n",
                        "    print(\"Please check your Kaggle API credentials and dataset key.\")\n",
                        "    raise"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Copy data to project directory\n",
                        "print(f\"Copying data from {DOWNLOAD_PATH} to {DATA_DIR}\")\n",
                        "\n",
                        "def copy_data_files(source_dir, target_dir):\n",
                        "    \"\"\"Copy all files from source to target directory.\"\"\"\n",
                        "    source_path = Path(source_dir)\n",
                        "    target_path = Path(target_dir)\n",
                        "    \n",
                        "    # Ensure target directory exists\n",
                        "    target_path.mkdir(parents=True, exist_ok=True)\n",
                        "    \n",
                        "    copied_files = []\n",
                        "    \n",
                        "    for file_path in source_path.rglob('*'):\n",
                        "        if file_path.is_file():\n",
                        "            # Maintain relative directory structure\n",
                        "            relative_path = file_path.relative_to(source_path)\n",
                        "            target_file_path = target_path / relative_path\n",
                        "            \n",
                        "            # Create parent directories if needed\n",
                        "            target_file_path.parent.mkdir(parents=True, exist_ok=True)\n",
                        "            \n",
                        "            # Copy file\n",
                        "            shutil.copy2(file_path, target_file_path)\n",
                        "            copied_files.append(target_file_path)\n",
                        "            print(f\"  ✓ Copied: {relative_path}\")\n",
                        "    \n",
                        "    return copied_files\n",
                        "\n",
                        "try:\n",
                        "    copied_files = copy_data_files(DOWNLOAD_PATH, DATA_DIR)\n",
                        "    print(f\"\\n✓ Successfully copied {len(copied_files)} files to {DATA_DIR}\")\n",
                        "except Exception as e:\n",
                        "    print(f\"❌ Error copying files: {e}\")\n",
                        "    raise"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Verify data integrity and provide summary\n",
                        "print(\"Verifying data integrity...\")\n",
                        "\n",
                        "def verify_data_integrity(data_dir):\n",
                        "    \"\"\"Verify that data was copied correctly and provide summary.\"\"\"\n",
                        "    data_path = Path(data_dir)\n",
                        "    \n",
                        "    if not data_path.exists():\n",
                        "        print(f\"❌ Data directory does not exist: {data_path}\")\n",
                        "        return False\n",
                        "    \n",
                        "    # Get all files\n",
                        "    all_files = list(data_path.rglob('*'))\n",
                        "    data_files = [f for f in all_files if f.is_file()]\n",
                        "    \n",
                        "    if not data_files:\n",
                        "        print(f\"❌ No files found in data directory: {data_path}\")\n",
                        "        return False\n",
                        "    \n",
                        "    print(f\"\\n📊 Data Summary for {SLUG}:\")\n",
                        "    print(f\"{'='*50}\")\n",
                        "    print(f\"Total files: {len(data_files)}\")\n",
                        "    \n",
                        "    # Analyze file types\n",
                        "    file_types = {}\n",
                        "    total_size = 0\n",
                        "    \n",
                        "    for file_path in data_files:\n",
                        "        file_ext = file_path.suffix.lower()\n",
                        "        file_size = file_path.stat().st_size\n",
                        "        \n",
                        "        if file_ext not in file_types:\n",
                        "            file_types[file_ext] = {'count': 0, 'size': 0}\n",
                        "        \n",
                        "        file_types[file_ext]['count'] += 1\n",
                        "        file_types[file_ext]['size'] += file_size\n",
                        "        total_size += file_size\n",
                        "    \n",
                        "    print(f\"Total size: {total_size / (1024**2):.2f} MB\")\n",
                        "    print(\"\\nFile types:\")\n",
                        "    for ext, info in file_types.items():\n",
                        "        ext_name = ext if ext else 'no extension'\n",
                        "        size_mb = info['size'] / (1024**2)\n",
                        "        print(f\"  {ext_name}: {info['count']} files ({size_mb:.2f} MB)\")\n",
                        "    \n",
                        "    # Try to load CSV files for basic validation\n",
                        "    csv_files = [f for f in data_files if f.suffix.lower() == '.csv']\n",
                        "    if csv_files:\n",
                        "        print(\"\\n📋 CSV File Preview:\")\n",
                        "        for csv_file in csv_files[:3]:  # Preview first 3 CSV files\n",
                        "            try:\n",
                        "                df = pd.read_csv(csv_file, nrows=5)  # Read only first 5 rows\n",
                        "                print(f\"\\n{csv_file.name}:\")\n",
                        "                print(f\"  Shape: {df.shape} (showing first 5 rows)\")\n",
                        "                print(f\"  Columns: {list(df.columns)}\")\n",
                        "                display(df.head())\n",
                        "            except Exception as e:\n",
                        "                print(f\"  ⚠️ Could not preview {csv_file.name}: {e}\")\n",
                        "    \n",
                        "    print(\"\\n✅ Data verification completed successfully!\")\n",
                        "    return True\n",
                        "\n",
                        "# Run verification\n",
                        "verification_success = verify_data_integrity(DATA_DIR)\n",
                        "\n",
                        "if verification_success:\n",
                        "    print(f\"\\n🎉 Setup completed successfully for project: {SLUG}\")\n",
                        "    print(\"\\nNext steps:\")\n",
                        "    print(f\"1. Create new notebooks in: {NOTEBOOK_DIR}\")\n",
                        "    print(\"2. Load shared variables with: %store -r\")\n",
                        "    print(\"3. Start your analysis!\")\n",
                        "else:\n",
                        "    print(\"\\n❌ Setup verification failed. Please check the errors above.\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"✓ Generated setup notebook: {notebook_path}")
    
    def run_setup(self) -> None:
        """Execute the complete setup process."""
        print(f"Starting setup for Data Science project: {self.slug}")
        print(f"Dataset: {self.dataset_key}")
        print(f"Git root: {self.git_root}")
        print("-" * 60)
        
        try:
            # Step 1: Create directories
            self.create_directory_structure()
            print()
            
            # Step 2: Generate setup notebook
            self.generate_setup_notebook()
            print()
            
            print("🎉 Project setup completed successfully!")
            print(f"\nTo initialize your project:")
            print(f"1. cd {self.slug_dirs['notebooks']}")
            print(f"2. jupyter notebook setup.ipynb")
            print(f"3. Run all cells in the setup notebook")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise


def main():
    """Main function to run the setup script."""
    # Example usage - modify these values for your specific project
    SLUG = "customer-churn-analysis"  # Your project identifier
    DATASET_KEY = "blastchar/telco-customer-churn"  # Kaggle dataset key
    
    # Initialize and run setup
    setup = DataScienceProjectSetup(
        slug=SLUG,
        dataset_key=DATASET_KEY,
        git_root=None  # Uses current directory if None
    )
    
    setup.run_setup()


if __name__ == "__main__":
    main()