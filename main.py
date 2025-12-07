import random
import time
import smtplib
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
import hashlib
from collections import defaultdict, deque
import pickle

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------
# Enhanced Device Configurations
# ---------------------------------------------------------
devices = {
    "lights": {"power": 60, "state": False, "last_used": None, "idle_timeout": 600, "uptime": 0, "category": "lighting", "owner": "family"},
    "hvac": {"power": 1500, "state": False, "last_used": None, "idle_timeout": 900, "uptime": 0, "category": "climate", "owner": "family"},
    "oven": {"power": 2000, "state": False, "last_used": None, "idle_timeout": 300, "uptime": 0, "category": "kitchen", "owner": "mom"},
    "refrigerator": {"power": 200, "state": True, "last_used": None, "idle_timeout": None, "uptime": 0, "category": "essential", "owner": "family"},
    "tv": {"power": 120, "state": False, "last_used": None, "idle_timeout": 1200, "uptime": 0, "category": "entertainment", "owner": "dad"},
    "computer": {"power": 200, "state": False, "last_used": None, "idle_timeout": 900, "uptime": 0, "category": "office", "owner": "kid"},
    "washing_machine": {"power": 500, "state": False, "last_used": None, "idle_timeout": 3600, "uptime": 0, "category": "laundry", "owner": "mom"},
}

# Historical usage with blockchain-like structure
usage_history = []
weekly_energy_log = []
daily_energy_log = defaultdict(list)
monthly_bills = []

# Financial settings
electricity_cost_per_kwh = 5
user_budget = 2000  # ₹2000 monthly budget
CARBON_PER_KWH = 0.85  # kg CO2 per kWh

# Auto schedules
user_preferences = {
    "hvac": {"schedule": [(8, 22)]},
    "lights": {"schedule": [(18, 23)]},
}

# ---------------------------------------------------------
# FEATURE 10: Budget Prediction & Carbon Tracker
# ---------------------------------------------------------
class CarbonTracker:
    def __init__(self):
        self.total_carbon_saved = 0
        self.carbon_history = []
        
    def calculate_carbon(self, energy_kwh):
        return energy_kwh * CARBON_PER_KWH
    
    def get_carbon_savings_today(self):
        today = datetime.now().date()
        if today in daily_energy_log:
            daily_energy = sum(daily_energy_log[today]) / 3600 / 1000  # Convert to kWh
            avg_daily = np.mean([sum(v) for v in daily_energy_log.values()]) / 3600 / 1000 if daily_energy_log else 0
            savings = avg_daily - daily_energy if avg_daily > 0 else 0
            return self.calculate_carbon(savings)
        return 0
    
    def get_equivalent_impact(self, carbon_kg):
        """Convert carbon savings to real-world equivalents"""
        equivalents = {
            "trees": carbon_kg / 21.77,  # kg CO2 absorbed by tree/year
            "cars": carbon_kg / 4600,    # kg CO2 from car/year
            "phones": carbon_kg / 50     # kg CO2 from phone charging/year
        }
        return equivalents

class BudgetPredictor:
    def __init__(self, monthly_budget=2000):
        self.monthly_budget = monthly_budget
        self.daily_budget = monthly_budget / 30
        self.alert_threshold = 0.8  # Alert at 80% of budget
        
    def predict_monthly_bill(self):
        """Predict final monthly bill based on current usage"""
        if not daily_energy_log:
            return 0
            
        current_month = datetime.now().month
        month_days = 30  # Simplified
        
        # Calculate average daily energy in kWh
        daily_energies = []
        for date, energies in daily_energy_log.items():
            if date.month == current_month:
                daily_kwh = sum(energies) / 3600 / 1000  # Convert W-sec to kWh
                daily_energies.append(daily_kwh)
        
        if not daily_energies:
            return 0
            
        avg_daily_kwh = np.mean(daily_energies)
        days_remaining = month_days - len(daily_energies)
        projected_kwh = (avg_daily_kwh * days_remaining) + sum(daily_energies)
        projected_bill = projected_kwh * electricity_cost_per_kwh
        
        return projected_bill
    
    def check_budget_alerts(self):
        """Check if we're exceeding budget and send alerts"""
        projected_bill = self.predict_monthly_bill()
        
        if projected_bill > self.monthly_budget * self.alert_threshold:
            overshoot_percent = ((projected_bill - self.monthly_budget) / self.monthly_budget) * 100
            
            if overshoot_percent > 10:
                alert_level = "⚠️ CRITICAL"
                action = "Aggressive optimization activated"
                auto_optimize_aggressive_mode()
            elif overshoot_percent > 5:
                alert_level = "⚠️ WARNING"
                action = "Moderate optimization activated"
            else:
                alert_level = "ℹ️ INFO"
                action = "Monitor usage"
            
            message = f"{alert_level}: Projected bill ₹{projected_bill:.0f} exceeds {self.alert_threshold*100:.0f}% of budget (₹{self.monthly_budget})"
            logging.warning(f"{message}. {action}")
            
            # Suggest savings tips
            suggest_energy_saving_tips(overshoot_percent)
            
            return True, projected_bill
        
        return False, projected_bill

def suggest_energy_saving_tips(overshoot_percent):
    """Provide context-aware energy saving suggestions"""
    tips = []
    
    # Analyze current usage patterns
    hour = datetime.now().hour
    total_power = calculate_energy_usage()
    
    if total_power > 2000:
        tips.append("High power draw detected. Consider staggering high-power appliance usage.")
    
    if hour >= 22 or hour < 6:
        tips.append("Night time: Use energy-saving mode on all devices.")
    
    # Check specific devices
    if devices.get("hvac", {}).get("state"):
        tips.append("HVAC running: Consider increasing temperature by 1°C to save 3-5% energy.")
    
    if devices.get("lights", {}).get("state") and hour < 18:
        tips.append("Lights on during daylight: Consider using natural light.")
    
    if tips:
        logging.info("💡 Energy Saving Tips:")
        for tip in tips:
            logging.info(f"   • {tip}")

def auto_optimize_aggressive_mode():
    """More aggressive energy optimization when budget is exceeded"""
    logging.info("🔧 AGGRESSIVE OPTIMIZATION MODE ACTIVATED")
    
    # Turn off non-essential devices
    for device, info in devices.items():
        if device not in ["refrigerator"] and info["state"]:
            info["state"] = False
            logging.info(f"   • Turned off {device} to save energy")
    
    # Adjust settings
    if "hvac" in devices:
        devices["hvac"]["idle_timeout"] = 300  # Shorter timeout
    
    # Send notification
    send_urgent_budget_alert()

def send_urgent_budget_alert():
    """Send urgent notification about budget exceedance"""
    # Could be SMS, push notification, or highlighted email
    logging.warning("🚨 URGENT: Monthly budget at risk! Taking aggressive energy saving measures.")

# ---------------------------------------------------------
# FEATURE 11 & 12: Gamification & Family Competition
# ---------------------------------------------------------
class EnergyGame:
    def __init__(self):
        self.points = defaultdict(int)
        self.levels = defaultdict(int)
        self.achievements = defaultdict(list)
        self.weekly_challenges = {}
        self.leaderboard = {}
        self.initialize_family_members()
        
    def initialize_family_members(self):
        """Initialize family members with their devices"""
        self.family_members = {
            "dad": {"devices": ["tv", "computer"], "points": 0, "level": 1},
            "mom": {"devices": ["oven", "washing_machine"], "points": 0, "level": 1},
            "kid": {"devices": ["lights", "tv"], "points": 0, "level": 1},
            "family": {"devices": ["hvac", "refrigerator"], "points": 0, "level": 1}
        }
        
    def award_points(self, member, action, energy_saved=0):
        """Award points for energy-saving actions"""
        points_map = {
            "off_peak_usage": 10,
            "turned_off_idle": 5,
            "optimized_schedule": 15,
            "manual_turn_off": 8,
            "weekly_challenge": 50,
            "budget_saver": 20,
            "carbon_hero": 25
        }
        
        points = points_map.get(action, 0)
        
        # Bonus points for significant energy savings
        if energy_saved > 100:  # More than 100W saved
            points += int(energy_saved / 50)
        
        self.family_members[member]["points"] += points
        
        # Check for level up
        old_level = self.family_members[member]["level"]
        new_level = min(10, 1 + (self.family_members[member]["points"] // 100))
        
        if new_level > old_level:
            self.family_members[member]["level"] = new_level
            achievement = f"Level {new_level} Reached!"
            self.achievements[member].append(achievement)
            logging.info(f"🎉 {member.capitalize()} leveled up to Level {new_level}!")
        
        # Check for special achievements
        self.check_achievements(member)
        
        return points
    
    def check_achievements(self, member):
        """Check and award special achievements"""
        member_data = self.family_members[member]
        points = member_data["points"]
        
        achievements = {
            100: "Energy Saver Bronze ⭐",
            500: "Energy Saver Silver ⭐⭐",
            1000: "Energy Saver Gold ⭐⭐⭐",
            2000: "Energy Master 🌟",
            5000: "Planet Hero 🌍"
        }
        
        for threshold, title in achievements.items():
            if points >= threshold and title not in self.achievements[member]:
                self.achievements[member].append(title)
                logging.info(f"🏆 {member.capitalize()} unlocked: {title}!")
    
    def calculate_member_usage(self):
        """Calculate energy usage for each family member"""
        member_usage = defaultdict(float)
        
        for device, info in devices.items():
            if info["state"]:
                owner = info.get("owner", "family")
                member_usage[owner] += info["power"]
        
        return member_usage
    
    def update_leaderboard(self):
        """Update and display weekly leaderboard"""
        member_usage = self.calculate_member_usage()
        
        # Convert to efficiency score (lower usage = higher score)
        for member in self.family_members:
            usage = member_usage.get(member, 0)
            score = max(0, 1000 - usage)  # Base score minus usage
            score += self.family_members[member]["points"] * 10  # Add points bonus
            
            self.leaderboard[member] = {
                "score": int(score),
                "usage": usage,
                "points": self.family_members[member]["points"],
                "level": self.family_members[member]["level"]
            }
        
        # Sort leaderboard
        sorted_leaderboard = sorted(self.leaderboard.items(), 
                                   key=lambda x: x[1]["score"], 
                                   reverse=True)
        
        return sorted_leaderboard
    
    def get_weekly_champion(self):
        """Determine weekly energy champion"""
        leaderboard = self.update_leaderboard()
        if leaderboard:
            champion, data = leaderboard[0]
            return champion, data["score"]
        return None, 0
    
    def display_game_stats(self):
        """Display game statistics in dashboard"""
        leaderboard = self.update_leaderboard()
        
        logging.info("\n🏆 FAMILY ENERGY CHALLENGE 🏆")
        logging.info("=" * 40)
        
        for rank, (member, data) in enumerate(leaderboard[:3], 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            logging.info(f"{medal} {member.upper():<8} | Score: {data['score']:>6} | "
                        f"Usage: {data['usage']:>5.0f}W | "
                        f"Points: {data['points']:>4} | Level: {data['level']}")
        
        if len(leaderboard) > 3:
            logging.info("...")
            for member, data in leaderboard[3:]:
                logging.info(f"   {member.upper():<8} | Score: {data['score']:>6} | "
                           f"Usage: {data['usage']:>5.0f}W")

# ---------------------------------------------------------
# FEATURE 13: Edge AI for Anomaly Detection
# ---------------------------------------------------------
class EdgeAIDetector:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.power_history = deque(maxlen=window_size)
        self.device_patterns = {}
        self.anomalies_detected = 0
        
    def update_power_history(self, current_power):
        """Update rolling window of power readings"""
        self.power_history.append(current_power)
        
        # Train simple anomaly detection when we have enough data
        if len(self.power_history) >= self.window_size:
            self.detect_anomalies()
    
    def detect_anomalies(self):
        """Detect anomalies in power usage patterns"""
        if len(self.power_history) < self.window_size:
            return
        
        # Simple statistical anomaly detection
        history_array = np.array(self.power_history)
        mean = np.mean(history_array)
        std = np.std(history_array)
        current = history_array[-1]
        
        # Check for spikes (3 sigma rule)
        if std > 0 and abs(current - mean) > 3 * std:
            self.anomalies_detected += 1
            anomaly_type = self.classify_anomaly(current, mean, std)
            
            logging.warning(f"🚨 ANOMALY DETECTED: {anomaly_type}")
            logging.warning(f"   Current: {current:.0f}W, Mean: {mean:.0f}W, Std: {std:.0f}W")
            
            # Take action based on anomaly type
            self.handle_anomaly(anomaly_type, current)
            
            return True
        
        return False
    
    def classify_anomaly(self, current, mean, std):
        """Classify the type of anomaly"""
        deviation = (current - mean) / std if std > 0 else 0
        
        if deviation > 3:
            if current > 3000:  # Very high power
                return "POWER SURGE - Possible multiple appliances on"
            else:
                return "UNUSUAL HIGH USAGE"
        elif deviation < -3:
            return "UNUSUAL LOW USAGE - Possible device failure"
        else:
            return "PATTERN DEVIATION"
    
    def handle_anomaly(self, anomaly_type, current_power):
        """Take appropriate action for detected anomaly"""
        actions = {
            "POWER SURGE": self.handle_power_surge,
            "UNUSUAL HIGH USAGE": self.handle_high_usage,
            "UNUSUAL LOW USAGE": self.handle_low_usage,
            "PATTERN DEVIATION": self.handle_pattern_deviation
        }
        
        for key, handler in actions.items():
            if key in anomaly_type:
                handler(current_power)
                break
    
    def handle_power_surge(self, current_power):
        """Handle power surge anomaly"""
        logging.warning("   🔌 Power surge detected! Checking for overload...")
        
        # Turn off non-essential devices
        priority_order = ["oven", "hvac", "washing_machine", "tv", "computer", "lights"]
        
        for device in priority_order:
            if device in devices and devices[device]["state"]:
                devices[device]["state"] = False
                logging.warning(f"   • Auto-off {device} to prevent overload")
                break
    
    def handle_high_usage(self, current_power):
        """Handle unusually high usage"""
        hour = datetime.now().hour
        
        if hour >= 22 or hour < 6:  # Night time
            logging.warning("   🌙 High usage during night - possible intrusion or forgetfulness")
            # Could trigger security camera recording
        
        # Check for idle devices
        optimize_energy()
    
    def handle_low_usage(self, current_power):
        """Handle unusually low usage"""
        logging.warning("   ⚡ Unusually low usage - checking device health")
        
        # Check if essential devices are off
        if devices.get("refrigerator", {}).get("state") == False:
            logging.error("   ❌ REFRIGERATOR IS OFF! This could cause food spoilage.")
            # Send urgent notification
    
    def handle_pattern_deviation(self, current_power):
        """Handle pattern deviation"""
        logging.info("   📊 Usage pattern deviation detected")
        # Log for further analysis, no immediate action
    
    def predict_next_hour_usage(self):
        """Simple prediction of next hour's energy usage"""
        if len(self.power_history) < 3:
            return calculate_energy_usage()
        
        # Simple moving average prediction
        return np.mean(list(self.power_history)[-3:])

# ---------------------------------------------------------
# Initialize Enhanced Systems
# ---------------------------------------------------------
carbon_tracker = CarbonTracker()
budget_predictor = BudgetPredictor(user_budget)
energy_game = EnergyGame()
edge_ai = EdgeAIDetector()

# ---------------------------------------------------------
# Enhanced Device Functions
# ---------------------------------------------------------
def track_energy_for_gamification(device, action, old_state, new_state):
    """Track device changes for gamification"""
    if old_state != new_state:
        owner = devices[device].get("owner", "family")
        
        if not new_state:  # Device turned OFF
            power_saved = devices[device]["power"]
            energy_game.award_points(owner, "manual_turn_off", power_saved)
            
            # Also award to family pool
            energy_game.award_points("family", "manual_turn_off", power_saved/2)

# ---------------------------------------------------------
# Enhanced Human Detection
# ---------------------------------------------------------
def detect_human():
    """Detects a human using camera; if camera fails, use fallback prediction."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            logging.warning("Camera not available — using fallback AI detection.")
            cap.release()
            return random.random() < 0.3   # 30% chance someone is around

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cap.release()
        
        detected = len(faces) > 0
        
        # Award points for human-aware actions
        if detected:
            energy_game.award_points("family", "off_peak_usage", 50)
        
        return detected

    except Exception:
        return random.random() < 0.2

# ---------------------------------------------------------
# Enhanced Simulation with Gamification
# ---------------------------------------------------------
def simulate_usage():
    current_time = datetime.now()

    for device, info in devices.items():
        if device == "refrigerator":
            continue

        probability = 0.15 if current_time.hour < 6 else 0.25

        if random.random() < probability:
            old_state = info["state"]
            info["state"] = not info["state"]
            info["last_used"] = current_time
            usage_history.append((current_time, device, info["state"]))
            
            # Track for gamification
            track_energy_for_gamification(device, "auto_toggle", old_state, info["state"])

        if info["state"]:
            info["uptime"] += 1

# ---------------------------------------------------------
# Enhanced Energy Calculations
# ---------------------------------------------------------
def calculate_energy_usage():
    total = sum(info["power"] for info in devices.values() if info["state"])
    
    # Update AI detector
    edge_ai.update_power_history(total)
    
    # Update daily log
    today = datetime.now().date()
    daily_energy_log[today].append(total)
    
    return total

# ---------------------------------------------------------
# Enhanced Auto-Schedule Logic with Gamification
# ---------------------------------------------------------
def check_schedules():
    current_hour = datetime.now().hour
    for device, prefs in user_preferences.items():
        for start, end in prefs["schedule"]:
            old_state = devices[device]["state"]
            new_state = (start <= current_hour < end)
            
            if old_state != new_state:
                devices[device]["state"] = new_state
                if new_state:
                    devices[device]["last_used"] = datetime.now()
                    energy_game.award_points("family", "optimized_schedule", 100)

# ---------------------------------------------------------
# Enhanced Device Health Score
# ---------------------------------------------------------
def device_health(device):
    """Returns rating from 1-5 based on uptime usage."""
    uptime = devices[device]["uptime"]
    if uptime < 10:
        return 5
    if uptime < 30:
        return 4
    if uptime < 60:
        return 3
    if uptime < 120:
        return 2
    return 1

# ---------------------------------------------------------
# Enhanced AI-Based Optimization with Gamification
# ---------------------------------------------------------
def optimize_energy():
    now = datetime.now()
    total_power = calculate_energy_usage()

    # 1. Idle shutdown per-device timers with points
    for dev, info in devices.items():
        if info["state"] and info["last_used"] and info["idle_timeout"]:
            if (now - info["last_used"]).total_seconds() > info["idle_timeout"]:
                old_state = info["state"]
                info["state"] = False
                power_saved = info["power"]
                
                owner = info.get("owner", "family")
                energy_game.award_points(owner, "turned_off_idle", power_saved)
                energy_game.award_points("family", "turned_off_idle", power_saved/2)
                
                logging.info(f"AI: Auto-off -> {dev} due to inactivity. +{power_saved} points")

    # 2. Overload protection
    if total_power > 2500:
        for dev in ["oven", "hvac", "lights"]:
            if devices[dev]["state"]:
                old_state = devices[dev]["state"]
                devices[dev]["state"] = False
                power_saved = devices[dev]["power"]
                
                owner = devices[dev].get("owner", "family")
                energy_game.award_points(owner, "budget_saver", power_saved)
                
                logging.warning(f"⚠ Overload! Turning off {dev}. +{power_saved} points")
                break

    # 3. AI-powered surge detection
    edge_ai.detect_anomalies()

# ---------------------------------------------------------
# Enhanced Prediction with AI
# ---------------------------------------------------------
def predict_energy_usage():
    if len(weekly_energy_log) < 7:
        return random.randint(1200, 2200)
    
    # Use AI prediction if available
    ai_prediction = edge_ai.predict_next_hour_usage()
    avg = sum(weekly_energy_log[-7:]) / 7
    
    # Blend AI prediction with historical average
    return (ai_prediction * 0.3 + avg * 0.7) * 1.05

# ---------------------------------------------------------
# Enhanced JSON Export with New Features
# ---------------------------------------------------------
def export_state():
    def clean_device(dev, name):
        return {
            "power": dev["power"],
            "state": dev["state"],
            "idle_timeout": dev["idle_timeout"],
            "uptime": dev["uptime"],
            "last_used": str(dev["last_used"]) if dev["last_used"] else None,
            "health": device_health(name),
            "owner": dev.get("owner", "family"),
            "category": dev.get("category", "general")
        }

    # Get game statistics
    leaderboard = energy_game.update_leaderboard()
    
    # Check budget alerts
    budget_warning, projected_bill = budget_predictor.check_budget_alerts()
    
    # Calculate carbon savings
    carbon_saved = carbon_tracker.get_carbon_savings_today()
    carbon_equivalents = carbon_tracker.get_equivalent_impact(carbon_saved)
    
    # Get weekly champion
    champion, champion_score = energy_game.get_weekly_champion()
    
    data = {
        "time": str(datetime.now()),
        "devices": {name: clean_device(info, name) for name, info in devices.items()},
        "total_power": calculate_energy_usage(),
        "history": [
            {
                "time": str(t),
                "device": d,
                "state": s
            }
            for (t, d, s) in usage_history[-10:]
        ],
        "gamification": {
            "leaderboard": [
                {
                    "member": member,
                    "score": data["score"],
                    "usage": data["usage"],
                    "points": data["points"],
                    "level": data["level"]
                }
                for member, data in leaderboard
            ],
            "weekly_champion": champion,
            "champion_score": champion_score,
            "achievements": dict(energy_game.achievements)
        },
        "budget": {
            "monthly_budget": user_budget,
            "projected_bill": round(projected_bill, 2),
            "budget_warning": budget_warning,
            "daily_budget": budget_predictor.daily_budget
        },
        "carbon": {
            "saved_today_kg": round(carbon_saved, 2),
            "equivalents": carbon_equivalents,
            "total_saved_kg": round(carbon_tracker.total_carbon_saved, 2)
        },
        "ai_insights": {
            "anomalies_detected": edge_ai.anomalies_detected,
            "next_hour_prediction": round(edge_ai.predict_next_hour_usage(), 0),
            "power_stability": "stable" if edge_ai.anomalies_detected == 0 else "unstable"
        }
    }

    with open("system_state.json", "w") as f:
        json.dump(data, f, indent=4)
    return data

# ---------------------------------------------------------
# Enhanced Email Report with New Features
# ---------------------------------------------------------
def send_email_report():
    sender = os.getenv("SENDER_EMAIL")
    receiver = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("EMAIL_PASSWORD")

    if not all([sender, receiver, password]):
        logging.error("Missing email credentials for report sending.")
        return

    total = calculate_energy_usage()
    cost = (total / 1000) * electricity_cost_per_kwh
    predicted = round(predict_energy_usage(), 2)
    
    # Get game stats
    leaderboard = energy_game.update_leaderboard()
    champion, _ = energy_game.get_weekly_champion()
    
    # Get budget info
    _, projected_bill = budget_predictor.check_budget_alerts()
    
    # Get carbon info
    carbon_saved = carbon_tracker.get_carbon_savings_today()
    carbon_eq = carbon_tracker.get_equivalent_impact(carbon_saved)

    body = f"""
📊 SMART SWITCHBOARD DAILY REPORT 📊
{datetime.now().strftime('%A, %B %d, %Y')}

🔌 ENERGY USAGE
• Current Power: {total}W
• Estimated Cost: ₹{round(cost, 2)}
• Predicted Tomorrow: {predicted}W
• Projected Monthly Bill: ₹{projected_bill:.2f}

🏆 FAMILY ENERGY CHALLENGE
• Weekly Champion: {champion.upper() if champion else "None yet"}
• Leaderboard:
"""
    
    for rank, (member, data) in enumerate(leaderboard[:3], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        body += f"  {medal} {member.upper()}: {data['score']} points (Level {data['level']})\n"
    
    body += f"""
🌍 ENVIRONMENTAL IMPACT
• Carbon Saved Today: {carbon_saved:.2f} kg CO2
• Equivalent to: {carbon_eq['trees']:.1f} trees 🌳
• That's like not driving {carbon_eq['cars']:.3f} cars for a year! 🚗

⚡ AI INSIGHTS
• Anomalies Detected: {edge_ai.anomalies_detected}
• System Health: {"✅ Excellent" if edge_ai.anomalies_detected == 0 else "⚠️ Needs Attention"}

💡 TIPS FOR TOMORROW
{suggest_energy_saving_tips_for_email()}
"""

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"Smart Switchboard Report - {datetime.now().date()}"
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        logging.info("📧 Enhanced email report sent.")
    except Exception as e:
        logging.error("Email failed:", e)

def suggest_energy_saving_tips_for_email():
    """Generate tips for email report"""
    tips = [
        "💡 Use natural light during daytime hours",
        "❄️ Set AC temperature to 24°C for optimal efficiency",
        "🔄 Run washing machine with full loads",
        "📺 Enable power-saving mode on all entertainment devices",
        "🔌 Unplug chargers when not in use"
    ]
    return "\n".join(tips)

# ---------------------------------------------------------
# Enhanced Dashboard Output
# ---------------------------------------------------------
def display_dashboard():
    logging.info("\n" + "="*60)
    logging.info("⚡ ENHANCED SMART DASHBOARD ⚡")
    logging.info(f"Time: {datetime.now()}")
    logging.info("="*60)
    
    # Device status
    for device, info in devices.items():
        health = device_health(device)
        logging.info(
            f"{device:<15} | {'ON' if info['state'] else 'OFF':<3} | "
            f"{info['power']:>4}W | Health: {health}★ | Owner: {info.get('owner', 'family'):<6}"
        )
    
    logging.info(f"\nTotal Power: {calculate_energy_usage()} W")
    
    # Display game stats
    energy_game.display_game_stats()
    
    # Display budget status
    budget_warning, projected_bill = budget_predictor.check_budget_alerts()
    if budget_warning:
        logging.warning(f"\n💰 BUDGET ALERT: Projected bill ₹{projected_bill:.0f} (Budget: ₹{user_budget})")
    else:
        logging.info(f"\n💰 Budget Status: OK (Projected: ₹{projected_bill:.0f})")
    
    # Display carbon impact
    carbon_saved = carbon_tracker.get_carbon_savings_today()
    logging.info(f"🌍 Carbon Saved Today: {carbon_saved:.2f} kg CO2")
    
    # Display AI insights
    logging.info(f"🤖 AI Insights: {edge_ai.anomalies_detected} anomalies detected")
    logging.info("="*60 + "\n")

# ---------------------------------------------------------
# Enhanced Main Loop
# ---------------------------------------------------------
def run_prototype():
    email_sent = False
    logging.info("🚀 ENHANCED Smart Switchboard System Starting...")
    logging.info("✨ Features: Gamification | Budget Alerts | Family Competition | Edge AI")
    
    # Initial game setup
    logging.info("🎮 Initializing Energy Game...")
    logging.info("👨‍👩‍👧‍👦 Family Members: Dad, Mom, Kid, Family")

    while True:
        hour = datetime.now().hour

        simulate_usage()
        check_schedules()
        optimize_energy()

        if detect_human():
            logging.info("👤 Human detected — keeping devices ON.")
        else:
            old_state = devices["lights"]["state"]
            devices["lights"]["state"] = False
            if old_state:
                energy_game.award_points("family", "carbon_hero", 60)
            logging.info("No human detected — turning off lights.")

        display_dashboard()
        export_state()

        if hour == 20 and not email_sent:
            send_email_report()
            email_sent = True

        if hour == 21:
            email_sent = False

        weekly_energy_log.append(calculate_energy_usage())

        time.sleep(5)

# ---------------------------------------------------------
# Start Enhanced System
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        run_prototype()
    except KeyboardInterrupt:
        logging.info("System stopped by user.")
        # Save game state before exit
        logging.info("Saving game statistics...")