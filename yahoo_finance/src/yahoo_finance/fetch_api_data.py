import yfinance as yf
from pydantic import BaseModel, Field

class StockPriceResult(BaseModel):
    """Result of the get stock price tool"""
    ticker: str = Field(..., description="The ticker symbol of the stock")
    price: float = Field(..., description="The current price of the stock")
    currency: str = Field(..., description="The currency of the stock price")

class StockPriceError(BaseModel):
    """Error result of the get stock price tool"""
    error: str = Field(..., description="The error message")

def get_stock_price(ticker_symbol: str) -> str:
    """Fetch the current price of stock given its ticker symbol"""
    try:
        stock = yf.Ticker(ticker_symbol.upper())
        price = stock.fast_info["last_price"]
        currency = stock.fast_info["currency"]

        result =  StockPriceResult(
            ticker=ticker_symbol.upper(),
            price=price,
            currency=currency
        )

        return result.model_dump_json()
    
    except Exception as e:
        print(f"Error fetching stock price for {ticker_symbol}: {e}")
        error = StockPriceError(
            error=str(e)
        )
        return error.model_dump_json()