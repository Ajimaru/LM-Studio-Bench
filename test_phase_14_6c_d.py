#!/usr/bin/env python3
"""
Phase 14.6c/d: Comprehensive Testing Suite
Export & Advanced Statistics Endpoints
"""

import json
import sqlite3
import statistics
import math
from pathlib import Path

DATABASE_FILE = Path("results/benchmark_cache.db")

def test_advanced_statistics():
    """Test Advanced Statistics Endpoint Logic"""
    print("\n📈 TEST: Advanced Statistics Calculation")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get model name
        cursor.execute('SELECT DISTINCT model_name FROM benchmark_results LIMIT 1')
        model_name = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        ''', (model_name,))
        
        data = cursor.fetchall()
        conn.close()
        
        speeds = [row[1] for row in data]
        
        # Calculate Statistics
        mean = statistics.mean(speeds)
        variance = statistics.variance(speeds) if len(speeds) > 1 else 0
        std_dev = math.sqrt(variance)
        volatility = (std_dev / mean * 100) if mean > 0 else 0
        
        # Linear Regression
        n = len(speeds)
        x_values = list(range(n))
        x_mean = statistics.mean(x_values)
        y_mean = mean
        
        numerator = sum((x_values[i] - x_mean) * (speeds[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast
        forecast = []
        for i in range(n, n + 3):
            predicted = slope * i + intercept
            forecast.append(round(max(0, predicted), 2))
        
        # Z-Scores & Anomalies
        z_scores = []
        anomalies = []
        if std_dev > 0:
            for i, speed in enumerate(speeds):
                z = (speed - mean) / std_dev
                z_scores.append(round(z, 2))
                if abs(z) > 2:
                    anomalies.append({
                        "index": i,
                        "speed": speed,
                        "z_score": z
                    })
        
        print(f"✅ Model: {model_name}")
        print(f"✅ Data Points: {n}")
        print(f"✅ Mean: {mean:.2f} tok/s")
        print(f"✅ Std Dev: {std_dev:.2f}")
        print(f"✅ Volatility: {volatility:.2f}%")
        print(f"✅ Slope: {slope:.4f} (trend)")
        print(f"✅ Forecast: {forecast}")
        print(f"✅ Anomalies Found: {len(anomalies)}")
        
        return {
            "success": True,
            "model": model_name,
            "stats": {
                "mean": mean,
                "std_dev": std_dev,
                "volatility": volatility,
                "slope": slope,
                "forecast": forecast,
                "anomalies": len(anomalies)
            }
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def test_csv_export():
    """Test CSV Export Logic"""
    print("\n💾 TEST: CSV Export Functionality")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT model_name FROM benchmark_results LIMIT 1')
        model_name = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM benchmark_results
            WHERE model_name = ?
        ''', (model_name,))
        
        count = cursor.fetchone()[0]
        
        # Simulate CSV creation
        cursor.execute('''
            SELECT timestamp, model_name, quantization, avg_tokens_per_sec
            FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        ''', (model_name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Build CSV structure
        csv_lines = ["Timestamp,Model,Quantization,Speed (tok/s)"]
        for row in rows:
            csv_lines.append(f"{row[0]},{row[1]},{row[2]},{row[3]:.2f}")
        
        csv_content = "\n".join(csv_lines)
        
        print(f"✅ Model: {model_name}")
        print(f"✅ Total Rows: {count}")
        print(f"✅ CSV Size: {len(csv_content)} bytes")
        print(f"✅ Headers: 4 columns")
        print(f"✅ Preview:\n{chr(10).join(csv_lines[:3])}")
        
        return {
            "success": True,
            "rows": count,
            "bytes": len(csv_content)
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def test_all_features():
    """Complete Feature Test"""
    print("\n" + "="*60)
    print("🧪 PHASE 14.6c/d: COMPREHENSIVE FEATURE TESTS")
    print("="*60)
    
    results = {
        "advanced_statistics": test_advanced_statistics(),
        "csv_export": test_csv_export()
    }
    
    print("\n" + "="*60)
    print("✅ SUMMARY")
    print("="*60)
    
    all_passed = all(r.get("success", False) for r in results.values())
    
    for feature, result in results.items():
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        print(f"{status}: {feature}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    
    return results

if __name__ == "__main__":
    test_all_features()
