import gradio as gr
import os
import sys
import json
import time
from typing import cast, Dict, Any, Tuple
from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction, Phase

# --- UI Theme & Constants ---
CSS = """
.dashboard-container { background: #0f172a; padding: 20px; border-radius: 15px; color: white; }
.traffic-grid { display: grid; grid-template-columns: repeat(3, 120px); grid-template-rows: repeat(3, 120px); gap: 10px; justify-content: center; margin: 20px 0; }
.road-cell { background: #334155; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; font-weight: bold; }
.intersection { background: #1e293b; border: 2px solid #64748b; }
.light { width: 25px; height: 25px; border-radius: 50%; margin: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.5); }
.light-red { background: #ef4444; box-shadow: 0 0 15px #ef4444; }
.light-green { background: #22c55e; box-shadow: 0 0 15px #22c55e; }
.queue-pill { background: #0ea5e9; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-top: 5px; }
.priority-alert { background: #f59e0b; color: #78350f; font-weight: bold; padding: 10px; border-radius: 8px; animation: pulse 2s infinite; text-align: center; margin: 10px 0; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
.metric-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 15px; text-align: center; }
.metric-val { font-size: 1.5em; color: #38bdf8; font-family: monospace; }
"""

# Global state for UI persistence
state = {
    "env": None,
    "last_obs": None,
    "history": []
}

def format_grid_html(obs: Dict[str, Any]) -> str:
    """Generate the dynamic HTML for the traffic grid."""
    if not obs: return "<p>Environment not initialized</p>"
    
    phase = obs.get("phase", "NS")
    p_active = obs.get("active_priority", False)
    p_app = obs.get("next_priority_approach", "NONE")
    
    # helper to check if a direction has a green light
    light_ns = "light-green" if phase == "NS" else "light-red"
    light_ew = "light-green" if phase == "EW" else "light-red"
    
    # helper for priority highlighting
    def cell_style(app: str):
        if p_active and p_app == app:
            return "border: 2px solid #f59e0b; background: #451a03;"
        return ""

    grid_html = f"""
    <div class="dashboard-container">
        {f'<div class="priority-alert">⚠️ EMERGENCY VEHICLE approaching from {p_app}!</div>' if p_active else ''}
        <div class="traffic-grid">
            <div class="empty"></div>
            <div class="road-cell" style="{cell_style('N')}">
                <span>North</span>
                <div class="light {light_ns}"></div>
                <div class="queue-pill">{obs.get('queue_north', 0)} vehicles</div>
            </div>
            <div class="empty"></div>
            
            <div class="road-cell" style="{cell_style('W')}">
                <span>West</span>
                <div class="light {light_ew}"></div>
                <div class="queue-pill">{obs.get('queue_west', 0)} vehicles</div>
            </div>
            <div class="road-cell intersection">
                <span style="font-size:0.7em; color:#94a3b8;">INTX</span>
                <div style="font-size:1.2em;">{phase}</div>
            </div>
            <div class="road-cell" style="{cell_style('E')}">
                <span>East</span>
                <div class="light {light_ew}"></div>
                <div class="queue-pill">{obs.get('queue_east', 0)} vehicles</div>
            </div>
            
            <div class="empty"></div>
            <div class="road-cell" style="{cell_style('S')}">
                <span>South</span>
                <div class="light {light_ns}"></div>
                <div class="queue-pill">{obs.get('queue_south', 0)} vehicles</div>
            </div>
            <div class="empty"></div>
        </div>
    </div>
    """
    return grid_html

def initialize_ui(task: str, seed: int):
    env = SmartAdaptiveTrafficSignalEnv(task_name=task, seed=int(seed))
    obs = env.reset()
    state["env"] = env
    state["last_obs"] = obs.model_dump()
    state["history"] = []
    
    metrics = env.get_metrics()
    return format_grid_html(state["last_obs"]), f"Ready - Task: {task}", 0, 0.0, f"Score: {env.evaluate():.4f}"

def run_step(phase: str):
    if state["env"] is None:
        return gr.update(), "Please reset first", 0, 0.0, "Score: 0.0"
    
    env = state["env"]
    action = TrafficAction(phase=cast(Phase, phase))
    obs, reward, done, info = env.step(action)
    state["last_obs"] = obs.model_dump()
    
    metrics = env.get_metrics()
    score = env.evaluate()
    
    status = "COMPLETED" if done else "RUNNING"
    return format_grid_html(state["last_obs"]), f"Step {obs.step} - Status: {status}", obs.step, reward.value, f"Score: {score:.4f}"

def run_auto_move():
    """Simple heuristic logic for the 'Auto' button in UI."""
    if state["env"] is None: return run_step("NS")
    
    obs = state["last_obs"]
    # Priority check
    if obs.get("active_priority"):
        app = obs.get("next_priority_approach")
        phase = "NS" if app in {"N", "S"} else "EW"
    else:
        ns = obs.get("queue_north", 0) + obs.get("queue_south", 0)
        ew = obs.get("queue_east", 0) + obs.get("queue_west", 0)
        phase = "NS" if ns >= ew else "EW"
    
    return run_step(phase)

def run_demo():
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Smart-Sync Traffic Hub") as demo:
        gr.Markdown("# 🚥 Smart-Sync Adaptive Traffic Hub")
        gr.Markdown("Real-time reinforcement learning simulation for emergency-first traffic management.")
        
        with gr.Row():
            with gr.Column(scale=2):
                grid_viz = gr.HTML(value="<p style='text-align:center;'>Initialize environment to see simulation</p>")
                
                with gr.Row():
                    phase_input = gr.Radio(["NS", "EW"], value="NS", label="Signal Phase Control")
                    step_btn = gr.Button("Manual Step", variant="primary")
                    auto_btn = gr.Button("💡 AI Suggestion", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Session Config")
                task_input = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Scenario")
                seed_input = gr.Number(42, label="Simulation Seed", precision=0)
                reset_btn = gr.Button("Reset Simulation")
                status_txt = gr.Textbox("Not Started", label="System Logs", interactive=False)
                
                gr.Separator()
                gr.Markdown("### 📊 Performance Monitor")
                with gr.Row():
                    step_gauge = gr.Number(label="Steps", value=0, interactive=False)
                    reward_gauge = gr.Number(label="Last Reward", value=0.0, interactive=False)
                score_display = gr.Markdown("## Score: 0.0000")

        gr.Separator()
        with gr.Accordion("Technical Specs & Legend", open=False):
            gr.Markdown("""
            ### Metrics & Reward Structure
            - **NS Phase**: North-South traffic moves.
            - **EW Phase**: East-West traffic moves.
            - **Emergency First**: Agents are penalized heavily for delaying ambulances.
            - **Capacity Scaling**: Served roads reduce capacity for normal vehicles when a priority vehicle passes.
            
            ### Legend
            - 🟢 **Green Light**: Directions being served.
            - 🔴 **Red Light**: Directions waiting.
            - ⚠️ **Orange Flow**: Active Emergency Vehicle.
            """)

        # Event Bindings
        reset_btn.click(initialize_ui, [task_input, seed_input], [grid_viz, status_txt, step_gauge, reward_gauge, score_display])
        step_btn.click(run_step, [phase_input], [grid_viz, status_txt, step_gauge, reward_gauge, score_display])
        auto_btn.click(run_auto_move, None, [grid_viz, status_txt, step_gauge, reward_gauge, score_display])

    return demo

if __name__ == "__main__":
    demo = run_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
