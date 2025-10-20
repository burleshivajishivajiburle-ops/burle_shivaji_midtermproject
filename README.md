## Project Overview
This project implements and compares three different algorithms for datasets to discover frequent itemsets and association rules from transaction data. 

## Algorithms Implemented
1. **Brute-Force Apriori** - Custom implementation using brute force approach
2. **MLxtend Apriori** - Using the MLxtend library's optimized Apriori algorithm  
3. **FP-Growth** - Using the MLxtend library's FP-Growth algorithm

## Project Structure
```
shivaji_data/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/                          # Transaction datasets
│   ├── Amazon_Transactions.csv    # Amazon book purchases
│   ├── BestBuy_Transactions.csv   # BestBuy electronics
│   ├── Custom_Transactions.csv    # Custom grocery items
│   ├── KMart_Transactions.csv     # KMart retail items
│   └── Nike_Transactions.csv      # Nike sportswear items
├── src/                           # Source code
│   └── pythoncode.py             # Main implementation
├── notebooks/                     # Jupyter notebooks
│   └── Pythoncode.ipynb          # Interactive analysis notebook
├── Report/                        # Project documentation
│   └── burle_shivaji_midtermproject.docx
└── venv/                         # Python virtual environment
```



## Features
- **Multiple Algorithm Support**: Compare performance of different frequent pattern mining algorithms
- **Configurable Parameters**: Adjustable minimum support and confidence thresholds
- **Performance Timing**: Execution time comparison between algorithms
- **Clean Output Format**: ASCII-friendly display with proper formatting
- **Error Handling**: Robust input validation and error management
- **Memory Efficient**: Optimized for large transaction datasets

## Dependencies
- pandas (2.3.3)
- mlxtend (0.23.4)
- numpy (2.3.4)
- scikit-learn (1.7.2)
- matplotlib (3.10.7)
- scipy (1.16.2)

## Usage

### Running the Python Script
```bash
python src/pythoncode.py
```

### Running the Jupyter Notebook
```bash
jupyter notebook notebooks/Pythoncode.ipynb
```

## How It Works
1. **Dataset Selection**: Choose from 5 available transaction datasets
2. **Parameter Configuration**: Set minimum support and confidence values
3. **Algorithm Execution**: Run all three algorithms sequentially
4. **Results Display**: View frequent itemsets and association rules
5. **Performance Comparison**: Compare execution times across algorithms

## Key Metrics
- **Support**: Frequency of itemset occurrence in transactions
- **Confidence**: Likelihood of consequent given antecedent in association rules
- **Execution Time**: Performance comparison between algorithms

## Sample Output
```
Running Shoe, Socks | Support: 0.75
Running Shoe, Socks -> Sweatshirts (Confidence: 0.85)
```


## Installation
1. Clone or download the project
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script or notebook

## Notes
- The project uses a virtual environment (`venv/`) for dependency management
- All algorithms produce identical results but with different performance characteristics
- The implementation includes memory-efficient generators for large datasets
- Unicode characters are properly handled for cross-platform compatibility