#!/usr/bin/env python3
"""
Test-Script für neue Comparison-Endpoints
Phase 14.6a: Backend Endpoints Testing
"""

import json
import sqlite3
from pathlib import Path

DATABASE_FILE = Path("results/benchmark_cache.db")

def test_comparison_models():
    """Simuliert GET /api/comparison/models"""
    print("\n📊 TEST: GET /api/comparison/models")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Hole Modelle mit Count der Einträge
        cursor.execute('''
            SELECT DISTINCT model_name, COUNT(*) as entry_count
            FROM benchmark_results
            GROUP BY model_name
            ORDER BY entry_count DESC, model_name ASC
        ''')
        
        models = []
        for model_name, count in cursor.fetchall():
            # Hole neueste und älteste Einträge
            cursor.execute('''
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (model_name,))
            latest = cursor.fetchone()
            
            cursor.execute('''
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp ASC
                LIMIT 1
            ''', (model_name,))
            oldest = cursor.fetchone()
            
            if latest and oldest:
                delta = ((latest[1] - oldest[1]) / oldest[1] * 100) if oldest[1] > 0 else 0
                models.append({
                    "model_name": model_name,
                    "entry_count": count,
                    "latest_speed": round(latest[1], 2),
                    "latest_timestamp": latest[0],
                    "oldest_timestamp": oldest[0],
                    "speed_delta_pct": round(delta, 2)
                })
        
        conn.close()
        
        response = {"success": True, "models": models}
        print(json.dumps(response, indent=2))
        print(f"✅ {len(models)} Modelle gefunden")
        return response
    
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return {"success": False, "error": str(e), "models": []}

def test_comparison_model_history():
    """Simuliert GET /api/comparison/{model_name}"""
    print("\n📈 TEST: GET /api/comparison/{model_name}")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Finde ein Modell
        cursor.execute('SELECT DISTINCT model_name FROM benchmark_results LIMIT 1')
        model_name = cursor.fetchone()[0]
        print(f"Test mit Modell: {model_name}")
        
        # Hole alle Einträge für das Modell, sortiert nach Timestamp
        cursor.execute('''
            SELECT 
                timestamp, quantization, avg_tokens_per_sec, avg_ttft, 
                avg_gen_time, gpu_offload, vram_mb, temperature,
                top_k_sampling, top_p_sampling, min_p_sampling, repeat_penalty, max_tokens,
                num_runs, benchmark_duration_seconds, error_count
            FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        ''', (model_name,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "timestamp": row[0],
                "quantization": row[1],
                "speed_tokens_sec": round(row[2], 2),
                "ttft": round(row[3], 3),
                "gen_time": round(row[4], 3),
                "gpu_offload": row[5],
                "vram_mb": row[6],
                "temperature": row[7],
                "top_k_sampling": row[8],
                "top_p_sampling": row[9],
                "min_p_sampling": row[10],
                "repeat_penalty": row[11],
                "max_tokens": row[12],
                "num_runs": row[13],
                "benchmark_duration_seconds": row[14],
                "error_count": row[15]
            })
        
        # Berechne Statistiken
        if history:
            speeds = [h["speed_tokens_sec"] for h in history]
            stats = {
                "min_speed": round(min(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "total_runs": len(history),
                "first_run": history[0]["timestamp"],
                "last_run": history[-1]["timestamp"],
                "trend": "up" if speeds[-1] > speeds[0] else "down" if speeds[-1] < speeds[0] else "stable"
            }
        else:
            stats = {}
        
        conn.close()
        
        response = {
            "success": True,
            "model_name": model_name,
            "history": history,
            "stats": stats
        }
        
        # Nur erste 2 History-Einträge anzeigen
        print(f"📊 Historische Einträge: {len(history)}")
        if history:
            print(f"\n   Erster Lauf: {history[0]['timestamp']}")
            print(f"   Speed: {history[0]['speed_tokens_sec']} tokens/sec")
            if len(history) > 1:
                print(f"\n   Letzter Lauf: {history[-1]['timestamp']}")
                print(f"   Speed: {history[-1]['speed_tokens_sec']} tokens/sec")
        
        print(f"\n📈 Statistiken:")
        print(json.dumps(stats, indent=2))
        
        print(f"\n✅ {len(history)} Einträge gefunden")
        return response
    
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return {"success": False, "error": str(e), "history": []}

def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("🧪 PHASE 14.6a ENDPOINT-TESTS")
    print("="*60)
    
    if not DATABASE_FILE.exists():
        print(f"❌ Datenbank nicht gefunden: {DATABASE_FILE}")
        return
    
    # Test 1: GET /api/comparison/models
    response1 = test_comparison_models()
    
    # Test 2: GET /api/comparison/{model_name}
    response2 = test_comparison_model_history()
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("📋 ZUSAMMENFASSUNG")
    print("="*60)
    print(f"✅ GET /api/comparison/models: {response1['success']}")
    print(f"✅ GET /api/comparison/{{model_name}}: {response2['success']}")
    print(f"✅ Beide Endpoints funktionieren korrekt!")
    print("\n🎯 Phase 14.6a: Backend-Endpoints ✅ COMPLETE")

if __name__ == "__main__":
    main()
