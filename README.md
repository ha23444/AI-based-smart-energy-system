⚡ Smart Switchboard System
"Intelligent Energy Management System with AI Optimization & Gamification"

A comprehensive smart energy management system that optimizes electricity usage through AI, gamification, and real-time monitoring. This project combines IoT, computer vision, machine learning, and modern web technologies to create an intelligent energy-saving solution.

https://img.shields.io/badge/python-3.8%252B-blue
https://img.shields.io/badge/flask-2.3.3-green
https://img.shields.io/badge/opencv-4.8.0-red
https://img.shields.io/badge/license-MIT-yellow

✨ Key Features
🔌 Smart Device Management
Control 7+ smart devices with real-time toggle switches

Individual device monitoring with health scores (1-5★ rating)

Bulk controls: Turn ALL devices ON/OFF with single click

Category-based device grouping (Lighting, Kitchen, Entertainment, etc.)

🤖 AI-Powered Optimization
Edge AI Anomaly Detection: Real-time power surge detection and prevention

Predictive Analytics: Next-hour energy usage prediction using moving averages

Intelligent Automation: Auto-shutdown idle devices based on timeout settings

Human Presence Detection: Camera-based facial recognition for occupancy-aware control

🎮 Gamification & Family Competition
Family Leaderboard: Compete with family members to save energy

Points System: Earn points for energy-saving actions (turning off devices, off-peak usage)

Achievements & Levels: Unlock badges and level up (Bronze ⭐ to Planet Hero 🌍)

Weekly Champions: Crown the top energy saver each week

💰 Budget & Environmental Tracking
Budget Predictor: Monthly bill projections with early warning alerts

Carbon Footprint Calculator: Real-time CO₂ savings tracking

Environmental Impact: Convert savings to tree equivalents 🌳

Smart Tips: Context-aware energy saving suggestions

📊 Real-time Dashboard
Modern Glass-morphism UI: Sleek, responsive dashboard with dark theme

Interactive Charts: Live power usage graphs and device distribution

Multi-tab Interface: Organized sections for devices, game, budget, AI insights

Mobile Responsive: Works seamlessly on all devices

🚀 Quick Start
Prerequisites
Python 3.8 or higher

Webcam (for human detection - optional)

Modern web browser

Installation
Clone the repository

bash
git clone https://github.com/your-username/smart-switchboard.git
cd smart-switchboard
Install dependencies

bash
pip install -r requirements.txt
Run the system

bash
python ui_server.py
Access the dashboard
Open your browser and navigate to:

text
http://localhost:5000
Dependencies
txt
Flask==2.3.3        # Web framework for dashboard
numpy==1.24.3       # Numerical computing for AI
pandas==2.0.3       # Data analysis and logging
opencv-python==4.8.0.74  # Computer vision for human detection
📁 Project Structure
text
smart-switchboard/
│
├── main.py              # Core backend system with AI features
├── ui_server.py         # Flask web server with API endpoints
├── dashboard.html       # Modern web dashboard interface
├── system_state.json    # Auto-generated system state file
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
🔧 Detailed Features
Device Management
7 Pre-configured Devices: Lights, HVAC, Oven, Refrigerator, TV, Computer, Washing Machine

Real-time Control: Instant toggle switches with visual feedback

Device Health: Uptime-based health scores (1-5 stars)

Usage History: Track when devices were last used

Power Monitoring: Real-time wattage display for each device

AI & Automation
Schedule-based Control: Automatic ON/OFF based on time schedules

Idle Detection: Auto-shutdown devices after inactivity timeout

Overload Protection: Automatic shutdown when power exceeds 2500W

Anomaly Detection: Statistical analysis to detect unusual power patterns

Human Detection: Face recognition to determine room occupancy

Gamification System
Family Members: Dad, Mom, Kid, Family (shared devices)

Points System:

Manual Turn OFF: 8 points

Auto Turn OFF: 5 points

Budget Saver: 20 points

Carbon Hero: 25 points

Level Progression: 10 levels based on points

Special Achievements: Bronze/Silver/Gold saver, Energy Master, Planet Hero

Budget & Environment
Monthly Budget: ₹2000 default (customizable)

Cost Calculation: ₹5 per kWh

Carbon Tracking: 0.85 kg CO₂ per kWh

Real-world Equivalents: Trees planted, cars not driven, phones not charged

Alert System: Warnings at 80% budget utilization

Dashboard Features
Live Clock: Real-time date and time display

Power Meter: Animated current power consumption

Device Cards: Individual controls with detailed info

Leaderboard: Family member rankings

Budget Progress Bar: Visual budget consumption

AI Insights: Anomaly detection and predictions

Activity Log: Recent device state changes

🖥️ Usage Guide
1. Device Control
Individual Control: Click toggle switches on device cards

Bulk Control: Use "Turn All ON/OFF" buttons in Quick Controls

AI Optimization: Click "AI Optimize" for automatic energy savings

Simulate: Generate random usage patterns for testing

2. Monitoring Features
Devices Tab: View and control all devices

Energy Game Tab: Check leaderboard and achievements

Budget & Carbon Tab: Track expenses and environmental impact

AI Insights Tab: View anomaly detection and predictions

Analytics Tab: See usage charts and activity history

3. System Automation
The system automatically:

Turns OFF idle devices after timeout

Prevents power overloads

Sends daily email reports at 8 PM

Adjusts based on human presence

Follows pre-set schedules

🔌 API Endpoints
Device Management
GET /api/devices - Get all devices and system status

POST /api/toggle - Toggle specific device state

POST /api/toggle_all - Turn all devices ON/OFF

POST /api/optimize - Trigger AI optimization

Gamification
GET /api/game_stats - Get leaderboard and achievements

GET /api/budget - Get budget information and alerts

GET /api/carbon - Get carbon savings data

GET /api/ai_insights - Get AI analytics and predictions

System Management
GET /api/system_stats - Comprehensive system statistics

POST /api/simulate_usage - Trigger manual simulation

🔧 Configuration
Device Settings (in main.py)
python
devices = {
    "lights": {
        "power": 60,                    # Watts
        "state": False,                 # Initial state
        "idle_timeout": 600,            # Auto-off after 10 minutes idle
        "category": "lighting",         # Device category
        "owner": "family"               # Family member assigned
    },
    # ... other devices
}
Budget Settings
python
user_budget = 2000                     # ₹2000 monthly budget
electricity_cost_per_kwh = 5           # ₹5 per kWh
CARBON_PER_KWH = 0.85                  # kg CO₂ per kWh
Schedules
python
user_preferences = {
    "hvac": {"schedule": [(8, 22)]},   # ON from 8 AM to 10 PM
    "lights": {"schedule": [(18, 23)]}, # ON from 6 PM to 11 PM
}
📊 Sample Dashboard View
text
⚡ ENHANCED SMART DASHBOARD ⚡
Time: 2024-02-23 20:15:00
============================================================
lights           | OFF |   60W | Health: 5★ | Owner: family
hvac             | ON  | 1500W | Health: 4★ | Owner: family
oven             | OFF | 2000W | Health: 5★ | Owner: mom
refrigerator     | ON  |  200W | Health: 3★ | Owner: family
tv               | OFF |  120W | Health: 5★ | Owner: dad
computer         | OFF |  200W | Health: 4★ | Owner: kid
washing_machine  | OFF |  500W | Health: 5★ | Owner: mom

Total Power: 1700 W

🏆 FAMILY ENERGY CHALLENGE 🏆
========================================
🥇 DAD     | Score:   1520 | Usage:  120W | Points:  52 | Level: 2
🥈 MOM     | Score:   1480 | Usage:  520W | Points:  48 | Level: 2
🥉 KID     | Score:   1450 | Usage:  200W | Points:  45 | Level: 1
   FAMILY  | Score:   1320 | Usage: 1760W | Points:  32 | Level: 1

💰 Budget Status: OK (Projected: ₹1560)
🌍 Carbon Saved Today: 2.15 kg CO2
🤖 AI Insights: 0 anomalies detected
📧 Email Reports
The system sends daily reports at 8 PM with:

Current power usage and estimated cost

Family energy challenge results

Environmental impact (carbon savings)

AI insights and system health

Energy saving tips for tomorrow

Email Configuration
Set environment variables for email:

bash
export SENDER_EMAIL="your-email@gmail.com"
export RECEIVER_EMAIL="recipient@example.com"
export EMAIL_PASSWORD="your-app-password"
🚨 Troubleshooting
Common Issues
Webcam not working for human detection

System falls back to random detection (30% chance)

Check camera permissions

Camera is optional - system works without it

Dashboard not loading

Ensure Flask server is running (python ui_server.py)

Check browser console for errors

Verify all dependencies are installed

Toggle switches not working

Check browser JavaScript console

Ensure backend is running and accessible

Refresh the dashboard page

Email reports not sending

Verify email credentials in environment variables

Check Gmail "Less secure app access" settings

Use App Password if 2FA is enabled

Logging
The system logs to console with detailed information:

Device state changes

AI optimizations

Gamification points

Anomaly detections

Email report status

🔮 Future Enhancements
Planned features for future versions:

Mobile App: Native iOS/Android applications

Voice Control: Integration with Alexa/Google Assistant

Weather Integration: Adjust HVAC based on weather forecasts

Solar Integration: Optimize for solar panel usage

Advanced ML: Deep learning for pattern recognition

Community Features: Compare with neighborhood averages

Smart Grid Integration: Participate in demand response programs

🤝 Contributing
We welcome contributions! Here's how you can help:

Fork the repository

Create a feature branch

bash
git checkout -b feature/amazing-feature
Commit your changes

bash
git commit -m 'Add amazing feature'
Push to the branch

bash
git push origin feature/amazing-feature
Open a Pull Request

Development Setup
bash
# Clone and setup
git clone https://github.com/your-username/smart-switchboard.git
cd smart-switchboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing

# Run tests
pytest tests/
📚 Learning Resources
Technologies Used
Python: Backend logic and AI algorithms

Flask: Web framework for API and server

OpenCV: Computer vision for human detection

NumPy: Numerical computing for anomaly detection

Chart.js: Interactive charts on dashboard

Bootstrap 5: Responsive UI framework

JavaScript: Frontend interactivity

Related Concepts
Energy Management Systems (EMS)

Internet of Things (IoT)

Edge AI Computing

Gamification in Sustainability

Behavioral Energy Efficiency

👨‍💻 Authors
Your Name - GitHub Profile

Acknowledgments
Icons by Font Awesome

Charts by Chart.js

UI inspiration from modern dashboard designs

Energy calculation formulas from environmental studies

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

