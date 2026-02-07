# NYC Housing Price Analysis

Analysis of NYC housing sales data. The starter notebook downloads the dataset and loads it into a Pandas DataFrame.

## Prerequisites

- **Python 3.8+**
- **pip**

## Environment setup (local)

### 1. Clone and enter the project

```bash
cd NYC_Housing_Price_Analysis
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# or on Windows:  venv\Scripts\activate
```

### 3. Install Jupyter support (for running the notebook)

```bash
pip install ipykernel
```

(Optional: install dependencies ahead of time so the notebook doesn’t need to.)

```bash
pip install gdown pandas
```

### 4. Run the notebook

- Open `Beginner_Starter_Notebook_NYC_Housing.ipynb` in **Cursor**, **VS Code**, or **Jupyter**.
- Select the kernel that uses this project’s venv (e.g. **Python 3.x (./venv)**).
- Run the first code cell. It will:
  - Install `gdown` and `pandas` if missing.
  - Download the NYC housing CSV into the `data/` folder.
  - Load the data into a DataFrame `df`.
- If the cell reported “restart the kernel to use updated packages,” restart the kernel and run the cell again.

After that, you can run the rest of the notebook for analysis and visualizations.

## Dataset location

- **Local:** CSV is saved under `data/` (e.g. `data/nyc_housing_base.csv`).
- **Databricks:** Uses the absolute path `/Workspace/Users/<your-email>/`. The notebook detects the environment automatically.

## Cloud (Databricks)

The same notebook can be run on Databricks without code changes. Set `WORKSPACE_USER` in the first cell to match your Databricks user folder if needed.
