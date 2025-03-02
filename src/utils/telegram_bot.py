"""
Telegram bot module for sending fall detection alerts.
"""

import time
import requests

class TelegramBot:
    """Telegram bot for sending alerts"""
    def __init__(self, token, chat_id):
        """
        Initialize the Telegram bot
        
        Args:
            token: Telegram bot token obtained from BotFather
            chat_id: Chat ID to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.last_alert_time = 0
        self.min_alert_interval = 120  # 2 minutes between alerts to prevent spam
        print(f"Telegram bot initialized for chat ID: {chat_id}")
        
    def send_alert(self, message):
        """
        Send alert message to Telegram chat
        
        Args:
            message: Message text to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        current_time = time.time()
        
        # Throttle messages to prevent spam
        if current_time - self.last_alert_time < self.min_alert_interval:
            print("Alert throttled (too soon after previous alert)")
            return False
            
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            response = requests.post(self.api_url, data=payload)
            if response.status_code == 200:
                self.last_alert_time = current_time
                print("Alert sent successfully")
                return True
            else:
                print(f"Failed to send alert: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending alert: {e}")
            return False