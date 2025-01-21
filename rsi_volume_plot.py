import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone

def calculate_rsi(price_series, period):
    if period <= 0:
        return pd.Series(index=price_series.index)
    
    delta = price_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100.0 - (100.0 / (1.0 + rs))

def weighted_rsi(data, periods):
    valid_periods = [p for p in periods if p > 0]
    rsi_df = pd.DataFrame(index=data.index)
    
    for period in valid_periods:
        rsi_values = calculate_rsi(data['Close'], period)
        if not isinstance(rsi_values, pd.Series):
            rsi_values = rsi_values.squeeze()
        
        volume = data['Volume'].squeeze()
        if not isinstance(volume, pd.Series):
            volume = volume.squeeze()
        
        numerator = (rsi_values * volume).rolling(window=period, min_periods=1).sum()
        denominator = volume.rolling(window=period, min_periods=1).sum().replace(0, np.finfo(float).eps)
        volume_weighted_rsi = numerator / denominator
        
        if not isinstance(volume_weighted_rsi, pd.Series):
            volume_weighted_rsi = volume_weighted_rsi.squeeze()
        
        rsi_df[f'RSI_{period}'] = rsi_values
        rsi_df[f'Weighted_RSI_{period}'] = volume_weighted_rsi
    
    return rsi_df

def plot_price_volume_rsi():
    try:
        # Hardcode the stock ticker and RSI periods
        stock_ticker = "AWX.SI"
        periods = [0, 7, 14]
        
        # Retrieve data from yfinance
        stock_data = yf.download(stock_ticker, period='1y')
        
        # Check if data is empty
        if stock_data.empty:
            print(f"No data available for the stock ticker: {stock_ticker}")
            return
        
        # Ensure Volume is 1D and numeric, handle potential NaN values
        if stock_data['Volume'].ndim > 1:
            stock_data['Volume'] = stock_data['Volume'].squeeze()
        stock_data['Volume'] = stock_data['Volume'].astype(float).fillna(0)
        
        # Ensure index is in datetime format
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)

        # Filter out non-positive periods
        valid_periods = [p for p in periods if p > 0]
        
        if not valid_periods:
            print("No valid RSI periods provided. Please enter positive periods.")
            return
        
        # Calculate RSI and weighted RSI
        rsi_data = weighted_rsi(stock_data, valid_periods)
             
        # Generate current date and time in Singapore timezone
        singapore_timezone = timezone('Asia/Singapore')
        now = datetime.now(singapore_timezone)
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 15))
        #fig.suptitle(f'{stock_ticker} - Price, Volume, RSI Analysis', fontsize=16)
        
        # Add date and time to the title
        fig.suptitle(f'AEM(AWX.SI) - Price, Volume, RSI Analysis - Generated(SG Time): {formatted_time}', fontsize=16)
        
        
        # Plot Price
        axs[0].plot(stock_data.index, stock_data['Close'], color='blue', label='Close Price')
        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        
        # Plot Volume with hardcoded linear scale
        volume_ax = axs[0].twinx()
        volume_ax.set_yscale('linear')  # Hardcoded to linear
        
        # Convert volume to a flat 1D numpy array
        volume_to_plot = stock_data['Volume'].to_numpy().flatten()
        
        # Plotting volume with custom style
        volume_ax.plot(stock_data.index, volume_to_plot, 'r-', alpha=0.5, label='Volume')
        volume_ax.fill_between(stock_data.index, volume_to_plot, 0, color='red', alpha=0.3)
        volume_ax.set_ylabel('Volume')
        volume_ax.legend(loc='upper right')
        
        # Plot Weighted RSI
        line_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_periods)))
        for i, period in enumerate(valid_periods):
            color = line_colors[i]
            axs[1].plot(rsi_data.index, rsi_data[f'Weighted_RSI_{period}'], 
                        label=f'Weighted RSI ({period})', color=color)
        
        axs[1].set_ylabel('Weighted RSI')
        axs[1].axhline(70, linestyle='--', color='r', alpha=0.5)
        axs[1].axhline(30, linestyle='--', color='g', alpha=0.5)
        axs[1].legend(loc='upper left')
        
        # Plot Regular RSI
        for i, period in enumerate(valid_periods):
            color = line_colors[i]
            axs[2].plot(rsi_data.index, rsi_data[f'RSI_{period}'], 
                        label=f'RSI ({period})', color=color)
        
        axs[2].set_ylabel('RSI')
        axs[2].axhline(70, linestyle='--', color='r', alpha=0.5)
        axs[2].axhline(30, linestyle='--', color='g', alpha=0.5)
        axs[2].legend(loc='upper left')
        
        # Add vertical dashed lines
        for ax in axs:
            for x in stock_data.index[::20]:
                ax.axvline(x, linestyle='--', color='gray', alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        
        # Save the chart instead of showing it
       
        ##plt.savefig('MZH_Chart.png')
        ##plt.close()  # Close the plot to free up memory

        # debug
        import os
        plt.show()
        plt.savefig('AEM_Chart.jpg', format='jpg', dpi=300, bbox_inches='tight')
        ##plt.savefig('AEM_Chart.png')
        plt.close()
        print(os.path.exists('AEM_Chart.png'))  # Should print True if the file exists
        print(os.path.abspath('AEM_Chart.png'))  # Shows the absolute path of where the file should be
    
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print("Detailed traceback:")
        traceback.print_exc()

# Call the function to generate the chart
if __name__ == "__main__":
    plot_price_volume_rsi()
