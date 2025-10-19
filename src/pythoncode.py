#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import time
import os
from typing import List, Tuple, Dict, Optional, Set, Any, Union

# Resolve project directories independent of current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

#   PERFORMANCE & UTILITY HELPERS

def _clean_item(value: Any) -> str:
    """Return an ASCII-friendly string representation of an item."""
    if value is None:
        return ""

    s = str(value)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": '-',
        "\u2014": '-',
        "\u00a0": ' ',
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Collapse consecutive whitespace
    return " ".join(s.split())

def _format_itemset_display(itemset: Union[Tuple[str, ...], List[str], Set[str], str]) -> str:
    """Format an itemset for display using comma-separated ASCII strings."""
    if isinstance(itemset, (list, tuple, set)):
        items = [_clean_item(item) for item in itemset]
        return ", ".join(items)
    return _clean_item(itemset)

def memory_efficient_itemset_generator(items: List[str], size: int):
    
    from itertools import combinations
    for itemset in combinations(items, size):
        yield itemset

def validate_transaction_format(transactions: List[List[str]]) -> bool:
    
    if not isinstance(transactions, list):
        return False
    
    for transaction in transactions:
        if not isinstance(transaction, list):
            return False
        for item in transaction:
            if not isinstance(item, str):
                return False
    
    return True

#       CONSTANTS & CONFIG

class Config:

    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR

    # Dataset configuration
    DATASETS = {
        1: "Amazon_Transactions.csv",
        2: "BestBuy_Transactions.csv", 
        3: "KMart_Transactions.csv",
        4: "Custom_Transactions.csv",
        5: "Nike_Transactions.csv",
    }
    
    DATASET_DESCRIPTIONS = {
        1: "Amazon_Transactions.csv",
        2: "BestBuy_Transactions.csv",
        3: "KMart_Transactions.csv", 
        4: "Custom_Transactions.csv",
        5: "Nike_Transactions.csv"
    }
    
    # Column names
    TRANSACTION_COLUMN = 'Transaction'
    TRANSACTION_ID_COLUMN = 'Transaction ID'
    
    # Validation constants
    MIN_SUPPORT_RANGE = (0.0, 1.0)
    MIN_CONFIDENCE_RANGE = (0.0, 1.0)
    VALID_DATASET_CHOICES = (1, 5)
    MAX_INPUT_LENGTH = 1
    
    # Display formatting
    DECIMAL_PRECISION = 2
    TIME_PRECISION = 4
    
    # Algorithm names
    BRUTE_FORCE_NAME = "BRUTE-FORCE APRIORI"
    MLXTEND_APRIORI_NAME = "MLXTEND APRIORI" 
    FP_GROWTH_NAME = "FP-GROWTH"
    
    # Timing comparison
    TIMING_TABLE_WIDTH = 40
    ALGORITHM_COL_WIDTH = 20
    TIME_COL_WIDTH = 18
    
    # Warning thresholds
    ZERO_SUPPORT_WARNING = True
    ZERO_CONFIDENCE_WARNING = True

#     DATASET LOADING

def load_transaction_dataset(choice: int) -> Optional[pd.DataFrame]:

    try:
        file_name = Config.DATASETS.get(choice)
        if not file_name:
            raise ValueError(f"Invalid choice: {choice}. Please select between {Config.VALID_DATASET_CHOICES[0]} and {Config.VALID_DATASET_CHOICES[1]}.")
        file_path = os.path.join(Config.DATA_DIR, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{file_name}' not found in '{Config.DATA_DIR}'.")
        
        # Load dataset with validation
        df = pd.read_csv(file_path)
        
        # Validate dataset structure
        if df.empty:
            raise ValueError(f"Dataset '{file_path}' is empty.")
        
        if Config.TRANSACTION_COLUMN not in df.columns:
            raise ValueError(f"Dataset '{file_path}' missing required '{Config.TRANSACTION_COLUMN}' column.")
        
        # Check for null values in Transaction column
        null_count = df[Config.TRANSACTION_COLUMN].isnull().sum()
        if null_count > 0:
            print(f"Warning: Found {null_count} null transactions. These will be removed.")
            df = df.dropna(subset=[Config.TRANSACTION_COLUMN])

        print(f"Successfully loaded dataset: {file_name}")
        print(f"Dataset contains {len(df)} transactions")
        return df

    except FileNotFoundError as e:
        print(f"File Error: {e}")
        print(f"Please ensure all dataset files are located in '{Config.DATA_DIR}'.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Dataset '{file_path}' is empty or corrupted.")
        return None
    except pd.errors.ParserError as e:
        print(f"Parse Error: Unable to read '{file_path}' - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return None

#   BRUTE-FORCE

def brute_force(transactions: List[List[str]], min_support: float, min_confidence: float) -> Tuple[List[Tuple], List[int], List[Tuple]]:
    
    # Convert transactions to sets for faster subset operations
    transaction_sets = [set(t) for t in transactions]
    n_transactions = len(transactions)
    
    # Get all unique items and create 1-itemsets
    unique_items = sorted(set(item for t in transactions for item in t))
    
    # Initialize with frequent 1-itemsets
    current_frequent = []
    frequent_patterns = []
    pattern_counts = []
    
    # Find frequent 1-itemsets
    item_counts = {}
    for item in unique_items:
        count = sum(1 for t in transaction_sets if item in t)
        support = count / n_transactions
        if support >= min_support:
            current_frequent.append((item,))
            frequent_patterns.append((item,))
            pattern_counts.append(count)
            item_counts[item] = count
    
    # Generate frequent k-itemsets using brute force pruning
    k = 2
    while current_frequent:
        # Generate candidates using frequent (k-1)-itemsets
        candidates = _generate_candidates(current_frequent, k)
        new_frequent = []
        
        for candidate in candidates:
            # Pruning: check if all (k-1)-subsets are frequent
            if _has_infrequent_subset(candidate, current_frequent):
                continue
                
            # Count support for this candidate
            count = sum(1 for t in transaction_sets if set(candidate).issubset(t))
            support = count / n_transactions
            
            if support >= min_support:
                new_frequent.append(candidate)
                frequent_patterns.append(candidate)
                pattern_counts.append(count)
        
        current_frequent = new_frequent
        k += 1
    
    # Generate association rules
    rules = _generate_optimized_rules(frequent_patterns, pattern_counts, transaction_sets, min_confidence)
    return frequent_patterns, pattern_counts, rules

def _generate_candidates(frequent_itemsets: List[Tuple], k: int) -> List[Tuple]:
    
    candidates = []
    n = len(frequent_itemsets)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Join two (k-1)-itemsets if they differ by only one item
            set1, set2 = set(frequent_itemsets[i]), set(frequent_itemsets[j])
            if len(set1.union(set2)) == k:
                candidate = tuple(sorted(set1.union(set2)))
                if candidate not in candidates:
                    candidates.append(candidate)
    
    return candidates

def _has_infrequent_subset(candidate: Tuple, frequent_itemsets: List[Tuple]) -> bool:
    
    frequent_set = set(frequent_itemsets)
    
    # Check all (k-1)-subsets of the candidate
    for i in range(len(candidate)):
        subset = candidate[:i] + candidate[i+1:]
        if subset not in frequent_set:
            return True
    return False

def _generate_optimized_rules(frequent_patterns: List[Tuple], pattern_counts: List[int], 
                            transaction_sets: List[Set[str]], min_confidence: float) -> List[Tuple]:
    
    rules = []
    
    # Create a mapping for fast lookup of pattern counts
    pattern_count_map = {pattern: count for pattern, count in zip(frequent_patterns, pattern_counts)}
    
    for pattern, count in zip(frequent_patterns, pattern_counts):
        if len(pattern) <= 1:
            continue
            
        # Generate all possible antecedents (non-empty proper subsets)
        for i in range(1, len(pattern)):
            for antecedent in combinations(pattern, i):
                # Get antecedent count from our mapping (if available) or calculate
                antecedent_count = pattern_count_map.get(antecedent)
                if antecedent_count is None:
                    antecedent_count = sum(1 for t in transaction_sets if set(antecedent).issubset(t))
                
                if antecedent_count == 0:
                    continue
                    
                confidence = count / antecedent_count
                if confidence >= min_confidence:
                    consequent = tuple(set(pattern) - set(antecedent))
                    rules.append((antecedent, consequent, confidence))
    
    return rules

# Backward compatibility function
def generate_rules(frequent_patterns: List[Tuple], pattern_counts: List[int], 
                  transactions: List[List[str]], min_confidence: float) -> List[Tuple]:
    
    transaction_sets = [set(t) for t in transactions]
    return _generate_optimized_rules(frequent_patterns, pattern_counts, transaction_sets, min_confidence)

def display_brute_force_results(patterns: List[Tuple], counts: List[int], 
                              transactions: List[List[str]], rules: List[Tuple], 
                              runtime: float) -> None:
    
    print(f"\n{Config.BRUTE_FORCE_NAME}")
    print()
    
    for pattern, count in zip(patterns, counts):
        support = count / len(transactions)
        item_str = _format_itemset_display(pattern)
        print(f"{item_str} | Support: {support:.{Config.DECIMAL_PRECISION}f}")
    
    print("\nAssociation Rules:")
    for antecedent, consequent, confidence in rules:
        # Use ASCII arrow to avoid encoding issues on some consoles
        antecedent_str = _format_itemset_display(antecedent)
        consequent_str = _format_itemset_display(consequent)
        print(f"{antecedent_str} -> {consequent_str} (Confidence: {confidence:.{Config.DECIMAL_PRECISION}f})")
    
    print(f"Runtime: {runtime:.{Config.TIME_PRECISION}f} seconds")

#  MLXTEND ALGORITHM PIPELINES

def _prepare_mlxtend_data(transactions: List[List[str]]) -> pd.DataFrame:
    
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

def _display_mlxtend_results(freq_items: pd.DataFrame, rules: pd.DataFrame, 
                           runtime: float, algorithm_name: str) -> None:
    
    print(f"\n{algorithm_name}")
    print()
    
    for _, row in freq_items.iterrows():
        itemset_list = list(row['itemsets'])
        support = row['support']
        item_str = _format_itemset_display(itemset_list)
        print(f"{item_str} | Support: {support:.{Config.DECIMAL_PRECISION}f}")
    
    print("\nAssociation Rules:")
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents']) 
        confidence = row['confidence']
        # Use ASCII arrow to avoid encoding issues on some consoles
        antecedent_str = _format_itemset_display(antecedents)
        consequent_str = _format_itemset_display(consequents)
        print(f"{antecedent_str} -> {consequent_str} (Confidence: {confidence:.{Config.DECIMAL_PRECISION}f})")
    
    print(f"Runtime: {runtime:.{Config.TIME_PRECISION}f} seconds")

def display_timing_comparison(brute_force_time: float, apriori_time: float, 
                            fp_growth_time: float) -> None:
    """Display a formatted timing comparison table for all algorithms."""
    
    
   
    print("TIMING COMPARISON")
    print()

    # Header
    print(f"{'Algorithm':<{Config.ALGORITHM_COL_WIDTH}} | {'Execution Time (s)':<{Config.TIME_COL_WIDTH}}")
    print("-" * Config.TIMING_TABLE_WIDTH)

    # Data rows (use configured time precision)
    print(f"{'Brute Force':<{Config.ALGORITHM_COL_WIDTH}} | {brute_force_time:<{Config.TIME_COL_WIDTH}.{Config.TIME_PRECISION}f}")
    print(f"{'Apriori':<{Config.ALGORITHM_COL_WIDTH}} | {apriori_time:<{Config.TIME_COL_WIDTH}.{Config.TIME_PRECISION}f}")
    print(f"{'FP-Growth':<{Config.ALGORITHM_COL_WIDTH}} | {fp_growth_time:<{Config.TIME_COL_WIDTH}.{Config.TIME_PRECISION}f}")
    print()

def _run_mlxtend_algorithm(transactions: List[List[str]], min_support: float, 
                         min_confidence: float, algorithm_func: Any, 
                         algorithm_name: str) -> float:
    """Run MLxtend algorithm and return execution time."""
    
    df_encoded = _prepare_mlxtend_data(transactions)
    
    start = time.time()
    freq_items = algorithm_func(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    end = time.time()
    
    runtime = end - start
    _display_mlxtend_results(freq_items, rules, runtime, algorithm_name)
    
    return runtime

#       APRIORI Method

def run_apriori_mlxtend(transactions: List[List[str]], min_support: float, 
                       min_confidence: float) -> float:
    """Run MLxtend Apriori algorithm and return execution time."""
    
    return _run_mlxtend_algorithm(transactions, min_support, min_confidence, 
                          apriori, Config.MLXTEND_APRIORI_NAME)

#       FP-GROWTH METHOD

def run_fp_growth(transactions: List[List[str]], min_support: float, 
                 min_confidence: float) -> float:
    """Run FP-Growth algorithm and return execution time."""
    
    return _run_mlxtend_algorithm(transactions, min_support, min_confidence, 
                          fpgrowth, Config.FP_GROWTH_NAME)

#    INPUT VALIDATION FUNCTIONS

def get_valid_dataset_choice() -> Optional[int]:
    
    print("AVAILABLE DATASETS (from Create Data part):")
    print()
    
    for key, description in Config.DATASET_DESCRIPTIONS.items():
        print(f"{key}. {description}")
    print()
    
    while True:
        try:
            user_input = input("\nEnter your dataset choice (1-5): ").strip()
            
            # Input validation - must be only 1 to enter
            if len(user_input) != Config.MAX_INPUT_LENGTH:
                print(f"Error: Please enter only ONE digit ({Config.VALID_DATASET_CHOICES[0]}-{Config.VALID_DATASET_CHOICES[1]})")
                continue
                
            choice = int(user_input)
            
            if choice < Config.VALID_DATASET_CHOICES[0] or choice > Config.VALID_DATASET_CHOICES[1]:
                print(f"Error: Choice must be between {Config.VALID_DATASET_CHOICES[0]} and {Config.VALID_DATASET_CHOICES[1]}")
                continue
                
            print(f"Selected: {Config.DATASET_DESCRIPTIONS[choice]}")
            return choice
            
        except ValueError as e:
            print(f"Error: Please enter a valid integer ({Config.VALID_DATASET_CHOICES[0]}-{Config.VALID_DATASET_CHOICES[1]})")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Exiting...")
            return None

def get_valid_support() -> Optional[float]:
    
    min_val, max_val = Config.MIN_SUPPORT_RANGE
    
    while True:
        try:
            support_input = input(f"\nEnter Minimum Support ({min_val} - {max_val}): ").strip()
            
            if not support_input:
                print("Error: Support value cannot be empty")
                continue
                
            min_support = float(support_input)
            
            if min_support < min_val or min_support > max_val:
                print(f"Error: Support must be between {min_val} and {max_val}")
                continue
                
            if min_support == 0.0 and Config.ZERO_SUPPORT_WARNING:
                print("Warning: Support of 0.0 may generate too many patterns")
                
            print(f"Minimum Support set to: {min_support}")
            return min_support
            
        except ValueError as e:
            print("Error: Please enter a valid decimal number (e.g., 0.2, 0.5)")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Exiting...")
            return None

def get_valid_confidence() -> Optional[float]:
    
    min_val, max_val = Config.MIN_CONFIDENCE_RANGE
    
    while True:
        try:
            confidence_input = input(f"Enter Minimum Confidence ({min_val} - {max_val}): ").strip()
            
            if not confidence_input:
                print("Error: Confidence value cannot be empty")
                continue
                
            min_confidence = float(confidence_input)
            
            if min_confidence < min_val or min_confidence > max_val:
                print(f"Error: Confidence must be between {min_val} and {max_val}")
                continue
                
            if min_confidence == 0.0 and Config.ZERO_CONFIDENCE_WARNING:
                print("Warning: Confidence of 0.0 may generate too many rules")
                
            print(f"Minimum Confidence set to: {min_confidence}")
            return min_confidence
            
        except ValueError as e:
            print("Error: Please enter a valid decimal number (e.g., 0.6, 0.8)")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Exiting...")
            return None

#        MAIN EXECUTION

def _process_transaction_data(df: pd.DataFrame) -> List[List[str]]:
    
    try:
        # Process transaction column - split by commas and strip whitespace
        df[Config.TRANSACTION_COLUMN] = df[Config.TRANSACTION_COLUMN].apply(
            lambda x: [_clean_item(item.strip()) for item in str(x).split(',') if item.strip()]
        )
        
        transactions = df[Config.TRANSACTION_COLUMN].tolist()
        
        # Validate processed transactions
        if not transactions:
            raise ValueError("No valid transactions found after processing")
        
        # Remove empty transactions
        transactions = [t for t in transactions if t and len(t) > 0]
        
        if not transactions:
            raise ValueError("All transactions are empty after cleaning")
        
        return transactions
        
    except Exception as e:
        raise ValueError(f"Error processing transaction data: {e}")

def main() -> None:
    
    # Display program header
    print("DATA MINING PROJECT")
    print()
    
    try:
        # Step 1: Get valid dataset choice (single digit input required)
        choice = get_valid_dataset_choice()
        if choice is None:
            return
            
        # Step 2: Load and validate selected dataset
        df = load_transaction_dataset(choice)
        if df is None:
            print("Failed to load dataset. Program terminating.")
            return

        # Step 3: Process transaction data with validation
        print("\nProcessing transaction data...")
        try:
            transactions = _process_transaction_data(df)
            print(f"Successfully processed {len(transactions)} transactions")
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Step 4: Get user-specified parameters with validation
        print(f"\nPARAMETER CONFIGURATION:")
        print()
        
        min_support = get_valid_support()
        if min_support is None:
            return
            
        min_confidence = get_valid_confidence()
        if min_confidence is None:
            return

        # Step 5: Execute algorithms with performance monitoring
        print(f"\nStarting analysis with Support={min_support}, Confidence={min_confidence}")
        print()

        # Initialize timing variables
        brute_force_time = 0.0
        apriori_time = 0.0
        fp_growth_time = 0.0

        # Brute Force Apriori Algorithm
        print("\nRunning Brute-Force Apriori Algorithm...")
        try:
            start = time.time()
            patterns, counts, rules = brute_force(transactions, min_support, min_confidence)
            end = time.time()
            brute_force_time = end - start
            display_brute_force_results(patterns, counts, transactions, rules, brute_force_time)
        except Exception as e:
            print(f"Error in Brute-Force Apriori: {e}")

        # MLxtend Apriori Algorithm  
        print("\nRunning MLxtend Apriori Algorithm...")
        try:
            apriori_time = run_apriori_mlxtend(transactions, min_support, min_confidence)
        except Exception as e:
            print(f"Error in MLxtend Apriori: {e}")

        # FP-Growth Algorithm
        print("\nRunning FP-Growth Algorithm...")
        try:
            fp_growth_time = run_fp_growth(transactions, min_support, min_confidence)
        except Exception as e:
            print(f"Error in FP-Growth: {e}")
        
        # Display timing comparison
        if brute_force_time > 0 and apriori_time > 0 and fp_growth_time > 0:
            display_timing_comparison(brute_force_time, apriori_time, fp_growth_time)
        
        # Success summary
        print("Analysis completed successfully!")
        print("All three algorithms have been executed.")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting gracefully...")
        print("Thank you for using the Market Basket Analysis tool!")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        print("Please check your input and try again.")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
