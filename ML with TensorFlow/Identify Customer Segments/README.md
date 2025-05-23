# Unsupervised Learning
## Project: Identify Customer Segments

Inside this directory you will find one Jupyter notebooks:

- **Identify_Customer_Segments.ipynb**: Complete project from the Udacity ML with TensorFlow course.

---

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](http://scikit-learn.org/stable/) (version 1.6.1 or later)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

If you need to upgrade scikit-learn, you can run:

```bash
python -m pip install --upgrade scikit-learn
```

To verify your scikit-learn version:

```python
import sklearn
print(f"The scikit-learn version is {sklearn.__version__})
```

### Run

In a terminal or command window, navigate to the top-level project directory `identify_customer_segmentes/` and run one of the following commands:

```bash
ipython notebook Identify_Customer_Segments.ipynb
```  
or
```bash
jupyter notebook Identify_Customer_Segments.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Project Overview

The analysis involves several steps:

- **Data Preprocessing**: Cleaning and preparing both demographic and customer data
- **Feature Transformation**: Applying techniques like one-hot encoding and feature scaling
- **Dimensionality Reduction**: Using Principal Component Analysis (PCA) to reduce feature space
- **Clustering**: Applying K-means clustering to identify customer segments
- **Segment Analysis**: Interpreting clusters and identifying core customer groups

### Data

The project uses two datasets:

- Demographics data for the general population of Germany
- Demographics data for customers of a mail-order company

Both datasets contain features related to demographic attributes, financial status, and various lifestyle indicators.

Note: Data has been removed due to copyright reasons.
