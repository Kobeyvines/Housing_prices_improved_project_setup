# 📸 Image Management Guide

Complete guide to managing and saving images in the House Prices ML project.

## Overview

All visualizations, plots, and generated images should be saved to the `images/` folder for consistency and organization.

## Directory Structure

```
images/
├── .gitkeep                  # Ensures folder is tracked by Git
├── README.md                 # This file
├── project_structure.png     # Project architecture diagram
└── (generated plots)         # EDA, model, and training visualizations
```

## Image Management Strategy

### What Gets Saved to `images/`

✅ **Save to images/**:
- Exploratory Data Analysis (EDA) plots
- Feature importance visualizations
- Correlation heatmaps
- Model performance plots
- Prediction vs actual comparisons
- Training history/curves
- Confusion matrices
- Distribution plots
- Any other ML visualizations

### What Doesn't Get Committed

❌ **Ignored in Git** (.gitignore rules):
- Generated `.png`, `.jpg`, `.pdf` files (too large for Git)
- Automated notebook plots
- Temporary visualizations

✅ **Committed to Git**:
- `README.md` (this documentation)
- `.gitkeep` (ensures folder exists)
- `project_structure.png` (important documentation)

## Using the ImageSaver Class

### Quick Start

```python
from src.visualization import ImageSaver

# Create image saver
saver = ImageSaver(images_dir="images")

# Save a matplotlib figure
fig, ax = plt.subplots()
ax.plot(data)
saver.save_figure(fig, "my_plot.png")  # Automatically saves to images/my_plot.png
```

### Available Methods

```python
# Save matplotlib figure
saver.save_figure(fig, "filename.png")

# Plot feature importance
saver.plot_feature_importance(importance_values, feature_names)

# Plot data distribution
saver.plot_distribution(data_series, "distribution.png")

# Plot correlation heatmap
saver.plot_correlation_heatmap(df, "correlations.png")

# Plot predictions vs actual
saver.plot_predictions_vs_actual(y_true, y_pred)

# Plot training history
saver.plot_training_history(train_loss, val_loss)

# Plot confusion matrix
saver.plot_confusion_matrix(y_true, y_pred)

# Plot metrics summary
saver.plot_metrics_summary({"Accuracy": 0.85, "Precision": 0.92})
```

## In Jupyter Notebooks

### Setup

```python
# Top of notebook
%matplotlib inline
import matplotlib.pyplot as plt
from src.visualization import ImageSaver

# Initialize saver (adjust path based on notebook location)
saver = ImageSaver(images_dir="../images")  # For notebooks/ folder
```

### Saving Plots

```python
# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_title("My Plot")

# Save it
saver.save_figure(fig, "my_analysis.png")
```

## In Python Scripts

### Entry Point Scripts

```python
# In entrypoint scripts
from src.visualization import ImageSaver

saver = ImageSaver(images_dir="images")

# Generate and save visualizations
saver.plot_feature_importance(importance, feature_names)
```

## Example: Complete Workflow

```python
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization import ImageSaver

# Initialize
saver = ImageSaver(images_dir="images", dpi=300)

# Load data
df = pd.read_csv("data/01-raw/train.csv")

# 1. Distribution analysis
saver.plot_distribution(df["SalePrice"], "eda/saleprice_distribution.png")

# 2. Correlation heatmap
numeric_df = df.select_dtypes(include=["number"])
saver.plot_correlation_heatmap(numeric_df, "eda/correlation_heatmap.png")

# 3. Feature importance (after training)
saver.plot_feature_importance(
    importance_values,
    feature_names,
    top_n=20,
    filename="model/feature_importance.png"
)

# 4. Model performance
saver.plot_predictions_vs_actual(
    y_test,
    y_pred,
    filename="model/predictions.png"
)

print("✅ All visualizations saved to images/")
```

## Image Specifications

**Default Settings**:
- **Format**: PNG (lossless, best for plots)
- **DPI**: 300 (print quality)
- **Bbox**: Tight (removes whitespace)
- **Background**: White
- **Quality**: Maximum

**Customization**:

```python
# Custom DPI and settings
saver = ImageSaver(images_dir="images", dpi=150)  # Lower DPI for web
fig.savefig("images/plot.png", dpi=150, bbox_inches="tight")
```

## Organizing Images by Category

For large projects, organize images into subdirectories:

```
images/
├── README.md
├── .gitkeep
├── eda/
│   ├── distribution_saleprice.png
│   ├── correlation_heatmap.png
│   └── missing_values.png
├── model/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── predictions_vs_actual.png
└── training/
    ├── training_history.png
    ├── metrics_summary.png
    └── learning_curves.png
```

**Creating subdirectories**:

```python
saver.save_figure(fig, "eda/my_plot.png")  # Creates eda/ if needed
saver.save_figure(fig, "model/metrics.png")  # Creates model/ if needed
```

## Git Configuration

### Why Images Aren't Committed

1. **File Size**: PNG/JPG files are large (50KB-500KB each)
2. **Binary Format**: Git isn't efficient with binary files
3. **Reproducibility**: Plots can be regenerated from code

### Committing Only Important Images

To track important documentation images:

```bash
# Allow specific images
git add images/project_structure.png
git add images/README.md
git commit -m "Add documentation images"
```

### Regenerating Images

Since images aren't committed, regenerate them locally:

```bash
# Run EDA notebook
jupyter notebook notebooks/EDA.ipynb

# Or run scripts
python scripts/generate_visualizations.py
```

## Best Practices

### 1. **Use Meaningful Filenames**
```python
✅ Good:
saver.save_figure(fig, "feature_importance_top20.png")
saver.save_figure(fig, "predictions_vs_actual_test.png")

❌ Bad:
saver.save_figure(fig, "plot1.png")
saver.save_figure(fig, "figure.png")
```

### 2. **Organize by Category**
```python
saver.save_figure(fig, "eda/distribution.png")
saver.save_figure(fig, "model/metrics.png")
saver.save_figure(fig, "training/history.png")
```

### 3. **Use High DPI**
```python
# For publications or reports
saver = ImageSaver(images_dir="images", dpi=300)
```

### 4. **Add Titles and Labels**
```python
fig, ax = plt.subplots()
ax.plot(data)
ax.set_title("Descriptive Title")
ax.set_xlabel("X-axis Label")
ax.set_ylabel("Y-axis Label")
saver.save_figure(fig, "plot.png")
```

### 5. **Use Descriptive Titles in Plots**
```python
# Include what, when, and where in titles
ax.set_title("Feature Importance - RandomForest Model - Validation Set")
```

## Troubleshooting

### Issue: Images not showing in Jupyter

**Solution**: Set matplotlib backend
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
import matplotlib.pyplot as plt
```

### Issue: High disk usage from images

**Solution**: Reduce DPI or use JPEG
```python
saver = ImageSaver(images_dir="images", dpi=100)  # Lower DPI
fig.savefig("images/plot.jpg", format='jpg', quality=85)  # JPEG format
```

### Issue: Images directory not created

**Solution**: Use ImageSaver (it creates automatically)
```python
# This creates images/ if it doesn't exist
saver = ImageSaver(images_dir="images")
```

## Integration with CI/CD

Images generated during CI/CD runs can be:

1. **Stored as artifacts** in GitHub Actions
2. **Uploaded to cloud storage** (S3, GCS, etc.)
3. **Attached to reports**
4. **Used in dashboards**

Example GitHub Actions step:
```yaml
- name: Upload images
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: visualizations
    path: images/
```

## Related Files

- `src/visualization.py` - ImageSaver class with plotting methods
- `src/utils.py` - save_plot() and save_image() utility functions
- `notebooks/EDA.ipynb` - Example notebook with visualizations
- `notebooks/Baseline.ipynb` - Example notebook with model visualizations

## Commands

```bash
# List all saved images
ls -lah images/

# Count images
find images -type f -name "*.png" | wc -l

# View recent images
ls -lt images/ | head -10

# Remove all generated images (keep only README and .gitkeep)
find images -type f ! -name "README.md" ! -name ".gitkeep" -delete
```

## Summary

✅ **All visualizations → images/ folder**
✅ **Use ImageSaver class for consistency**
✅ **Organize by category (eda/, model/, etc.)**
✅ **Use meaningful filenames**
✅ **Don't commit generated images to Git**
✅ **Commit documentation images manually**

Happy plotting! 📈
