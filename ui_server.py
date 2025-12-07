from flask import Flask, jsonify, request, render_template
from threading import Thread
import time
import main
import json
from datetime import datetime
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/api/devices")
def api_devices():
    # Export and get current state
    data = main.export_state()
    return jsonify(data)

@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    device = request.json.get("device")
    if device in main.devices:
        # Track for gamification
        old_state = main.devices[device]["state"]
        main.devices[device]["state"] = not main.devices[device]["state"]
        main.devices[device]["last_used"] = datetime.now()
        
        # Log the action
        main.usage_history.append((datetime.now(), device, main.devices[device]["state"]))
        
        # Award points for manual action (turning OFF saves energy)
        if main.devices[device].get("owner") and not main.devices[device]["state"]:
            power_saved = main.devices[device]["power"]
            main.energy_game.award_points(
                main.devices[device]["owner"], 
                "manual_turn_off",
                power_saved
            )
            print(f"🎮 Points awarded to {main.devices[device]['owner']} for turning off {device}")
        
        # Update uptime
        if main.devices[device]["state"]:
            main.devices[device]["uptime"] += 1
        
        # Export updated state
        main.export_state()
        
        return jsonify({
            "success": True,
            "new_state": main.devices[device]["state"],
            "device": device,
            "power": main.devices[device]["power"]
        })
    return jsonify({"success": False, "message": "Device not found"})

@app.route("/api/toggle_all", methods=["POST"])
def api_toggle_all():
    state = request.json.get("state", False)
    
    for device in main.devices:
        if device != "refrigerator":  # Don't toggle essential devices
            main.devices[device]["state"] = state
            main.devices[device]["last_used"] = datetime.now()
            
            # Log the action
            main.usage_history.append((datetime.now(), device, state))
            
            # Award points for turning OFF
            if not state and main.devices[device].get("owner"):
                power_saved = main.devices[device]["power"]
                main.energy_game.award_points(
                    main.devices[device]["owner"], 
                    "manual_turn_off",
                    power_saved
                )
    
    main.export_state()
    
    return jsonify({
        "success": True,
        "message": f"All devices turned {'ON' if state else 'OFF'}",
        "state": state
    })

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Trigger energy optimization"""
    try:
        # Call optimization function
        main.optimize_energy()
        
        # Award points to family for optimization
        main.energy_game.award_points("family", "optimized_schedule", 200)
        
        # Export state
        main.export_state()
        
        return jsonify({
            "success": True,
            "message": "Energy optimization applied successfully!"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Optimization failed: {str(e)}"
        })

@app.route("/api/game_stats")
def api_game_stats():
    """Get gamification statistics"""
    leaderboard = main.energy_game.update_leaderboard()
    champion, score = main.energy_game.get_weekly_champion()
    
    return jsonify({
        "leaderboard": [
            {"member": member, **data} for member, data in leaderboard
        ],
        "weekly_champion": champion,
        "champion_score": score,
        "achievements": dict(main.energy_game.achievements)
    })

@app.route("/api/budget")
def api_budget():
    """Get budget information"""
    warning, projected = main.budget_predictor.check_budget_alerts()
    
    return jsonify({
        "monthly_budget": main.user_budget,
        "projected_bill": projected,
        "budget_warning": warning,
        "daily_budget": main.budget_predictor.daily_budget
    })

@app.route("/api/carbon")
def api_carbon():
    """Get carbon savings data"""
    carbon_saved = main.carbon_tracker.get_carbon_savings_today()
    equivalents = main.carbon_tracker.get_equivalent_impact(carbon_saved)
    
    return jsonify({
        "saved_today_kg": carbon_saved,
        "equivalents": equivalents,
        "total_saved_kg": main.carbon_tracker.total_carbon_saved
    })

@app.route("/api/ai_insights")
def api_ai_insights():
    """Get AI insights"""
    return jsonify({
        "anomalies_detected": main.edge_ai.anomalies_detected,
        "next_hour_prediction": main.edge_ai.predict_next_hour_usage(),
        "power_stability": "stable" if main.edge_ai.anomalies_detected == 0 else "unstable"
    })

@app.route("/api/system_stats")
def api_system_stats():
    """Get comprehensive system statistics"""
    data = main.export_state()
    
    # Calculate additional stats
    active_devices = sum(1 for device in main.devices.values() if device["state"])
    total_devices = len(main.devices)
    total_power = main.calculate_energy_usage()
    
    # Calculate average health
    health_sum = 0
    count = 0
    for device in main.devices:
        health = main.device_health(device)
        health_sum += health
        count += 1
    
    avg_health = health_sum / count if count > 0 else 0
    
    stats = {
        "active_devices": active_devices,
        "total_devices": total_devices,
        "total_power": total_power,
        "device_health_avg": avg_health,
        "last_updated": datetime.now().isoformat()
    }
    
    return jsonify({**data, **stats})

@app.route("/api/simulate_usage", methods=["POST"])
def api_simulate_usage():
    """Trigger manual simulation"""
    main.simulate_usage()
    main.export_state()
    
    return jsonify({
        "success": True,
        "message": "Usage simulation completed"
    })

def backend_thread():
    main.run_prototype()

def ui_thread():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Smart Switchboard System...")
    print("✨ Features: Gamification | Budget Alerts | Family Competition | Edge AI")
    print("🌐 Dashboard available at: http://localhost:5000")
    
    # Start backend in separate thread
    Thread(target=backend_thread, daemon=True).start()
    time.sleep(2)  # Give backend time to initialize
    
    # Start Flask server
    ui_thread()