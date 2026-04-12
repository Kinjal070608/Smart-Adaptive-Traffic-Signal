import os
import sys
import numpy as np
from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction

def run_fixed_cycle(task_name: str, cycle_len: int = 6) -> dict:
    env = SmartAdaptiveTrafficSignalEnv(task_name=task_name, seed=42)
    obs = env.reset()
    done = False
    
    total_q = []
    p_waits = []
    
    while not done:
        # Fixed cycle: swap every cycle_len steps
        phase = "NS" if (obs.step // cycle_len) % 2 == 0 else "EW"
        action = TrafficAction(phase=phase)
        obs, reward, done, info = env.step(action)
        
        total_q.append(obs.queue_north + obs.queue_east + obs.queue_south + obs.queue_west)
        if obs.active_priority:
            p_waits.append(obs.priority_wait)
            
    metrics = env.get_metrics()
    return {
        "avg_q": np.mean(total_q),
        "p_wait": np.mean(p_waits) if p_waits else 0,
        "throughput": metrics["total_departed"],
        "score": env.evaluate()
    }

def run_smart_agent(task_name: str, min_green_time: int = 3) -> dict:
    env = SmartAdaptiveTrafficSignalEnv(task_name=task_name, seed=42)
    obs = env.reset()
    done = False
    
    total_q = []
    p_waits = []
    current_phase = "NS"
    time_in_phase = 0
    
    while not done:
        # Smart Heuristic with minimum green time to prevent thrashing
        if obs.active_priority:
            suggested_phase = "NS" if obs.next_priority_approach in {"N", "S"} else "EW"
        else:
            ns = obs.queue_north + obs.queue_south
            ew = obs.queue_east + obs.queue_west
            suggested_phase = "NS" if ns >= ew else "EW"
            
        # Only switch if suggested phase is different AND we've been in current phase long enough
        if suggested_phase != current_phase and time_in_phase >= min_green_time:
            current_phase = suggested_phase
            time_in_phase = 0
        else:
            time_in_phase += 1
            
        action = TrafficAction(phase=current_phase)
        obs, reward, done, info = env.step(action)
        
        total_q.append(obs.queue_north + obs.queue_east + obs.queue_south + obs.queue_west)
        if obs.active_priority:
            p_waits.append(obs.priority_wait)
            
    metrics = env.get_metrics()
    return {
        "avg_q": np.mean(total_q),
        "p_wait": np.mean(p_waits) if p_waits else 0,
        "throughput": metrics["total_departed"],
        "score": env.evaluate()
    }

def get_mean(data):
    return sum(data) / len(data) if data else 0

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    
    f_res = [run_fixed_cycle(t) for t in tasks]
    s_res = [run_smart_agent(t) for t in tasks]
    
    print("--- FIXED CYCLE ---")
    print(f"Avg Queue: {get_mean([r['avg_q'] for r in f_res]):.2f}")
    print(f"Avg P-Wait: {get_mean([r['p_wait'] for r in f_res]):.2f}")
    print(f"Avg Throughput: {get_mean([r['throughput'] for r in f_res]):.2f}")
    print(f"Avg Score: {get_mean([r['score'] for r in f_res]):.2f}")
    
    print("\n--- SMART AGENT ---")
    print(f"Avg Queue: {get_mean([r['avg_q'] for r in s_res]):.2f}")
    print(f"Avg P-Wait: {get_mean([r['p_wait'] for r in s_res]):.2f}")
    print(f"Avg Throughput: {get_mean([r['throughput'] for r in s_res]):.2f}")
    print(f"Avg Score: {get_mean([r['score'] for r in s_res]):.2f}")
