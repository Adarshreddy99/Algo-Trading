import os
import gspread
from google.oauth2.service_account import Credentials
from logs.logging import logger

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
SPREADSHEET_ID = os.getenv("GOOGLE_SHEET_ID")

def connect_to_sheet():
    creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID)
    return sheet

def ensure_worksheet(sheet, title, headers):
    """
    Opens a worksheet by title; creates and adds headers if not existing or empty.
    Forces header update if headers don't match exactly.
    """
    try:
        worksheet = sheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=title, rows=1000, cols=len(headers))
        worksheet.append_row(headers)
        logger.info(f"Created worksheet tab '{title}' with headers {headers}")
        return worksheet

    # Check if headers exist and match
    existing_values = worksheet.get_all_values()
    needs_header_update = False
    
    if not existing_values or len(existing_values) == 0:
        needs_header_update = True
        logger.info(f"No existing data in '{title}', adding headers")
    elif len(existing_values[0]) != len(headers) or existing_values[0] != headers:
        needs_header_update = True
        logger.info(f"Headers mismatch in '{title}'. Expected: {headers}, Found: {existing_values[0] if existing_values else 'None'}")
    
    if needs_header_update:
        # Clear existing data and add new headers
        worksheet.clear()
        worksheet.append_row(headers)
        logger.info(f"Updated headers in worksheet '{title}' with {headers}")
    else:
        logger.info(f"Headers in '{title}' are already correct")
        
    return worksheet

def log_trade_signal(sheet, rows):
    """Log trade signals with both strategy and ML model recommendations plus indicator values"""
    headers = ["Date", "Stock Symbol", "Strategy Signal", "ML Signal", "Price", "RSI", "20-DMA", "50-DMA"]
    ws = ensure_worksheet(sheet, "Trade Log", headers)
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    logger.info(f"Logged {len(rows)} trade signal rows with indicator values")

def update_pnl_summary(sheet, rows):
    """Update P&L summary with daily P&L calculations"""
    headers = ["Date", "Stock Symbol", "P&L %"]
    ws = ensure_worksheet(sheet, "P&L Summary", headers)
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    logger.info(f"Updated 'P&L Summary' with {len(rows)} rows")

def update_win_ratio(sheet, win_ratios: dict):
    """Update win ratios for each stock"""
    headers = ["Stock Symbol", "Win Ratio"]
    ws = ensure_worksheet(sheet, "Win Ratio", headers)
    
    # Get existing data
    records = ws.get_all_records()
    existing = {r["Stock Symbol"]: idx+2 for idx, r in enumerate(records)}
    
    # Update or append win ratios
    for stock, ratio in win_ratios.items():
        if stock in existing:
            ws.update_cell(existing[stock], 2, ratio)
        else:
            ws.append_row([stock, ratio])
    
    logger.info(f"Updated 'Win Ratio' for {len(win_ratios)} stocks")