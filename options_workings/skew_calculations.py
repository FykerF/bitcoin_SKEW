import numpy as np
import pandas as pd

def et1(f0, k0) -> int:
    return -1 * (1 + np.log(f0/k0) - f0/k0)

def et2(f0, k0):
    return 2 * (np.log(k0/f0) * (f0/k0 -1) + 0.5 * (np.log(k0/f0))**2)

def et3(f0, k0):
    return 3 * ((np.log(k0/f0))**2) * (1/3 * np.log(k0/f0) - 1 + f0/k0)

def dis_factor(rf_rate: float, ttm: float) -> float:
    return np.exp(-rf_rate * ttm/255)

def forward_index_level(option_data: pd.DataFrame, rf_rate: float) -> float:
    current_min = 1000000
    strike_midquote = None
    dte_midquote = None
    underlying = option_data['UNDERLYING_PRICE'].iloc[0]
    strike_line = option_data["STRIKE"].drop_duplicates()
    
    for strike in strike_line:
        slicy = option_data[option_data["STRIKE"] == strike][["OPTION_RIGHT","BID_PRICE","ASK_PRICE","DTE","STRIKE"]] 
        if 'call' in slicy["OPTION_RIGHT"].values and 'put' in slicy["OPTION_RIGHT"].values:
            put = slicy[slicy["OPTION_RIGHT"] == 'put']
            call = slicy[slicy["OPTION_RIGHT"] == 'call']
            
            # Calculate mid prices and extract scalar values
            put_mid = ((put["BID_PRICE"].iloc[0] + put['ASK_PRICE'].iloc[0]) / 2)
            call_mid = ((call["BID_PRICE"].iloc[0] + call['ASK_PRICE'].iloc[0]) / 2)
            
            # Calculate midquote difference
            midquote_diff = (call_mid - put_mid) * underlying
            
            if midquote_diff < current_min:
                current_min = midquote_diff
                strike_midquote = strike
                dte_midquote = slicy['DTE'].iloc[0]
            
    # Calculate forward index level
    d_factor = dis_factor(rf_rate, 30.125)
    print(d_factor)
    forward_level = d_factor * midquote_diff + strike_midquote
    
    print(f"\nSelected strike: {strike_midquote}")
    print(f"Min difference: {current_min}")
    print(f"Forward level: {forward_level}")
    print('--------------')
    
    return forward_level, strike_midquote

def eliminate_strikes(option_data: pd.DataFrame, strike_midquote: float, rf_rate: float) -> pd.DataFrame:
    """
    Eliminate strikes according to CBOE SKEW methodology:
    1. Eliminate strikes with zero bid
    2. For puts: eliminate strikes < K0 where two consecutive zero bids occur
    3. For calls: eliminate strikes > K0 where two consecutive zero bids occur
    
    Parameters:
    option_data: DataFrame with option data including BID_PRICE, STRIKE, OPTION_RIGHT
    strike_midquote: The strike at which |Call - Put| is minimized (K0)
    rf_rate: Risk-free rate
    
    Returns:
    DataFrame with filtered option data suitable for SKEW calculation
    """
    # K0 is the strike_midquote - the strike where |Call - Put| is minimized
    k0 = strike_midquote
    
    print(f"K0 (strike midquote): {k0}")
    
    # Separate puts and calls
    puts = option_data[option_data['OPTION_RIGHT'] == 'put'].copy()
    calls = option_data[option_data['OPTION_RIGHT'] == 'call'].copy()
    
    # Sort puts by strike descending (from K0 down)
    puts = puts.sort_values('STRIKE', ascending=False)
    
    # Sort calls by strike ascending (from K0 up)
    calls = calls.sort_values('STRIKE', ascending=True)
    
    # For puts: eliminate strikes < K0 where two consecutive zero bids occur
    valid_puts = []
    consecutive_zero_bids = 0
    
    for idx, row in puts.iterrows():
        if row['STRIKE'] > k0:
            # Keep all puts with strikes >= K0
            valid_puts.append(idx)
        else:
            # For strikes < K0, check for consecutive zero bids
            if row['BID_PRICE'] == 0:
                consecutive_zero_bids += 1
                if consecutive_zero_bids >= 2:
                    # Stop including puts once we hit 2 consecutive zero bids
                    break
            else:
                consecutive_zero_bids = 0
                valid_puts.append(idx)
    
    # For calls: eliminate strikes > K0 where two consecutive zero bids occur
    valid_calls = []
    consecutive_zero_bids = 0
    
    for idx, row in calls.iterrows():
        if row['STRIKE'] <= k0:
            # Keep all calls with strikes <= K0
            valid_calls.append(idx)
        else:
            # For strikes > K0, check for consecutive zero bids
            if row['BID_PRICE'] == 0:
                consecutive_zero_bids += 1
                if consecutive_zero_bids >= 2:
                    # Stop including calls once we hit 2 consecutive zero bids
                    break
            else:
                consecutive_zero_bids = 0
                valid_calls.append(idx)
    
    # Combine valid puts and calls
    valid_indices = valid_puts + valid_calls
    filtered_data = option_data.loc[valid_indices]
    
    # Remove any remaining options with zero bid
    filtered_data = filtered_data[filtered_data['BID_PRICE'] > 0]
    
    print(f"\nOriginal options: {len(option_data)}")
    print(f"After filtering: {len(filtered_data)}")
    print(f"Valid puts: {len([i for i in valid_indices if i in puts.index])}")
    print(f"Valid calls: {len([i for i in valid_indices if i in calls.index])}")
    
    # Sort by strike for easier viewing
    filtered_data = filtered_data.sort_values('STRIKE')
    
    return filtered_data, k0

def calculate_delta_k(strikes):
    """Calculate ΔK for each strike"""
    strikes = sorted(strikes)
    delta_k = {}
    
    for i, strike in enumerate(strikes):
        if i == 0:  # First (lowest) strike
            delta_k[strike] = strikes[i+1] - strikes[i]
        elif i == len(strikes) - 1:  # Last (highest) strike
            delta_k[strike] = strikes[i] - strikes[i-1]
        else:  # Middle strikes
            delta_k[strike] = (strikes[i+1] - strikes[i-1]) / 2
    
    return pd.Series(delta_k)

def calcualte_p1(cleared_strikes:pd.DataFrame,dis_factor:float,epsilon):
    
    # Get the price for the day
    price_t: float = cleared_strikes['UNDERLYING_PRICE'].iloc[0]
    
    # First find midquotes
    midquotes: pd.Series = cleared_strikes[['BID_PRICE','ASK_PRICE']].sum(axis = 1)/2 * price_t
    
    # Reverse strike squared
    rss: pd.Series  = 1/(cleared_strikes['STRIKE'])**2
    
    # Calculate deltas of strikes
    delta_k_series = calculate_delta_k(cleared_strikes['STRIKE'])
    # Map delta_k values to the DataFrame index using the strike values
    delta_k = cleared_strikes['STRIKE'].map(delta_k_series)
    
    # Compute whole thing 
    product: pd.Series = rss * midquotes * delta_k
    
    p1: float = dis_factor*(-1 * product.sum()) + epsilon
    
    
    return p1

def calculate_p2(cleared_strikes:pd.DataFrame, dis_factor:float, epsilon, forward_price:float):
    
    # Get the price for the day
    price_t: float = cleared_strikes['UNDERLYING_PRICE'].iloc[0]
    
    # First find midquotes
    midquotes: pd.Series = cleared_strikes[['BID_PRICE','ASK_PRICE']].sum(axis = 1)/2 * price_t
    
    # Calculate 2/K^2 term
    two_over_k_squared: pd.Series = 2/(cleared_strikes['STRIKE'])**2
    
    # Calculate (1 - ln(K/F0)) term
    ln_term: pd.Series = 1 - np.log(cleared_strikes['STRIKE']/forward_price)
    
    # Calculate deltas of strikes
    delta_k_series = calculate_delta_k(cleared_strikes['STRIKE'])
    # Map delta_k values to the DataFrame index using the strike values
    delta_k = cleared_strikes['STRIKE'].map(delta_k_series)
    
    # Compute whole thing
    product: pd.Series = two_over_k_squared * ln_term * midquotes * delta_k
    
    p2: float = dis_factor * product.sum() + epsilon
    
    return p2

def calculate_p3(cleared_strikes:pd.DataFrame, dis_factor:float, epsilon, forward_price:float):
    
    # Get the price for the day
    price_t: float = cleared_strikes['UNDERLYING_PRICE'].iloc[0]
    
    # First find midquotes
    midquotes: pd.Series = cleared_strikes[['BID_PRICE','ASK_PRICE']].sum(axis = 1)/2 * price_t
    
    # Calculate 3/K^2 term
    three_over_k_squared: pd.Series = 3/(cleared_strikes['STRIKE'])**2
    
    # Calculate ln(K/F0) term
    ln_k_over_f0: pd.Series = np.log(cleared_strikes['STRIKE']/forward_price)
    
    # Calculate {2ln(K/F0) - ln^2(K/F0)} term
    ln_combination: pd.Series = 2 * ln_k_over_f0 - ln_k_over_f0**2
    
    # Calculate deltas of strikes
    delta_k_series = calculate_delta_k(cleared_strikes['STRIKE'])
    # Map delta_k values to the DataFrame index using the strike values
    delta_k = cleared_strikes['STRIKE'].map(delta_k_series)
    
    # Compute whole thing
    product: pd.Series = three_over_k_squared * ln_combination * midquotes * delta_k
    
    p3: float = dis_factor * product.sum() + epsilon
    
    return p3


def calculate_S(day:pd.DataFrame, rf_rate: float) -> float:
    # Get forward index level and strike midquote
    f0, strike_midquote = forward_index_level(day, rf_rate)
    
    # Eliminate strikes according to CBOE methodology
    filtered_options, k0 = eliminate_strikes(day, strike_midquote, rf_rate)
    
    # Get DTE and calculate discount factor
    dte = filtered_options['DTE'].iloc[0]
    ttm = dte  # Time to maturity in days
    discount = dis_factor(rf_rate, ttm)
    
    # Calculate epsilon values
    epsilon1 = et1(f0, k0)
    epsilon2 = et2(f0, k0)
    epsilon3 = et3(f0, k0)
    
    # Calculate P1, P2, P3
    p1 = calcualte_p1(filtered_options, discount, epsilon1)
    p2 = calculate_p2(filtered_options, discount, epsilon2, f0)
    p3 = calculate_p3(filtered_options, discount, epsilon3, f0)
    
    # Calculate S using the formula: S = (P3 - 3*P1*P2 + 2*P1^3) / (P2 - P1^2)^(3/2)
    numerator = p3 - 3*p1*p2 + 2*(p1**3)
    denominator = (p2 - p1**2)**(3/2)
    
    s = numerator / denominator
    
    # Calculate SKEW = 100 - 10*S
    skew = 100 - 10*s
    
    '''print(f"\nCalculated values:")
    print(f"F0: {f0}")
    print(f"K0: {k0}")
    print(f"P1: {p1}")
    print(f"P2: {p2}")
    print(f"P3: {p3}")
    print(f"S: {s}")
    print(f"SKEW: {skew}")'''
    
    return skew


def find_nearest_dte_options(day_data: pd.DataFrame, target_dte: float = 30.0) -> tuple:
    """
    Find options with DTEs nearest to target_dte from above and below.
    
    Parameters:
    day_data: DataFrame with all options for a given day
    target_dte: Target DTE (default 30 days)
    
    Returns:
    tuple: (near_term_df, next_term_df, near_dte, next_dte) or (exact_df, None, exact_dte, None) if exact match exists
    """
    # Get unique DTEs
    unique_dtes = sorted(day_data['DTE'].unique())
    
    # Check if we have exact 30 DTE
    if target_dte in unique_dtes:
        exact_dte_df = day_data[int(day_data['DTE']) == target_dte].copy()
        return exact_dte_df, None, target_dte, None
    
    # Find nearest DTEs from below and above
    below_dtes = [dte for dte in unique_dtes if dte < target_dte]
    above_dtes = [dte for dte in unique_dtes if dte > target_dte]
    
    # Check if there can be toleretable dte above or below
    if not below_dtes:
        next_dte = min(above_dtes)
        if abs(next_dte - target_dte) <= 10:
            next_term_df = day_data[day_data['DTE'] == next_dte].copy()
            return next_term_df, None, next_dte, None
        else:
            raise ValueError(f"Cannot find DTEs both below and above {target_dte}. Available DTEs: {unique_dtes}")
    elif not above_dtes:
        near_dte = max(below_dtes)
        if abs(near_dte - target_dte) <= 10:
            near_term_df = day_data[day_data['DTE'] == near_dte].copy()
            return near_term_df, None, near_dte, None
        else:
            raise ValueError(f"Cannot find DTEs both below and above {target_dte}. Available DTEs: {unique_dtes}")

    
    # Get the closest DTE from below (near-term)
    near_dte = max(below_dtes)
    # Get the closest DTE from above (next-term)
    next_dte = min(above_dtes)
    
    # Create dataframes for each term
    near_term_df = day_data[day_data['DTE'] == near_dte].copy()
    next_term_df = day_data[day_data['DTE'] == next_dte].copy()
    
    print(f"Target DTE: {target_dte}")
    print(f"Near-term DTE: {near_dte} ({len(near_term_df)} options)")
    print(f"Next-term DTE: {next_dte} ({len(next_term_df)} options)")
    
    return near_term_df, next_term_df, near_dte, next_dte

def calculate_weighted_skew(day_data: pd.DataFrame, rf_rate: float, target_dte: float = 30.0) -> float:
    """
    Calculate weighted SKEW when exact target DTE is not available.
    
    Parameters:
    day_data: DataFrame with all options for a given day
    rf_rate: Risk-free rate
    target_dte: Target DTE (default 30 days)
    
    Returns:
    float: Weighted SKEW value
    """
    # Find nearest DTEs and create dataframes
    near_term_df, next_term_df, near_dte, next_dte = find_nearest_dte_options(day_data, target_dte)
    
    # If we have exact DTE, just calculate SKEW directly
    if next_term_df is None:
        return calculate_S(near_term_df, rf_rate)
    
    # Calculate SKEW for both terms
    print("\nCalculating near-term SKEW...")
    skew_near = calculate_S(near_term_df, rf_rate)
    
    print("\nCalculating next-term SKEW...")
    skew_next = calculate_S(next_term_df, rf_rate)
    
    # Calculate weights based on time to expiration
    # Weight formula from CBOE paper: w = (T2 - T30)/(T2 - T1)
    weight_near = (next_dte*1440 - target_dte*1440) / (next_dte*1440 - near_dte*1440)
    weight_next = 1-weight_near
    
    # Calculate weighted SKEW
    weighted_skew = weight_near * skew_near + weight_next * skew_next
    
    
    print(f"Weighted SKEW: {weighted_skew:.2f}")
    return weighted_skew