import gradio as gr
import os
import sys
from typing import cast, Dict, Any, List
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
.sparkline-container { display: flex; align-items: flex-end; height: 60px; gap: 2px; background: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155; }
.spark-bar { background: #38bdf8; width: 6px; border-radius: 2px 2px 0 0; }
.spark-label { color: #94a3b8; font-size: 0.7em; margin-bottom: 5px; }
"""

# Global state
state = {"env": None, "last_obs": None, "history": []}

def format_grid_html(obs: Dict[str, Any]) -> str:
    if not obs: return "<p>Environment not initialized</p>"
    phase = obs.get("phase", "NS")
    p_active = obs.get("active_priority", False)
    p_app = obs.get("next_priority_approach", "NONE")
    light_ns = "light-green" if phase == "NS" else "light-red"
    light_ew = "light-green" if phase == "EW" else "light-red"
    def cell_style(app: str):
        if p_active and p_app == app: return "border: 2px solid #f59e0b; background: #451a03;"
        return ""
    grid_html = f'<div class="dashboard-container">{"<div class=\'priority-alert\'>⚠️ EMERGENCY VEHICLE approaching from " + p_app + "!</div>" if p_active else ""}<div class="traffic-grid"><div class="empty"></div><div class="road-cell" style="{cell_style("N")}"><span>North</span><div class="light {light_ns}"></div><div class="queue-pill">{obs.get("queue_north", 0)} vehicles</div></div><div class="empty"></div><div class="road-cell" style="{cell_style("W")}"><span>West</span><div class="light {light_ew}"></div><div class="queue-pill">{obs.get("queue_west", 0)} vehicles</div></div><div class="road-cell intersection"><span style="font-size:0.7em; color:#94a3b8;">INTX</span><div style="font-size:1.2em;">{phase}</div></div><div class="road-cell" style="{cell_style("E")}"><span>East</span><div class="light {light_ew}"></div><div class="queue-pill">{obs.get("queue_east", 0)} vehicles</div></div><div class="empty"></div><div class="road-cell" style="{cell_style("S")}"><span>South</span><div class="light {light_ns}"></div><div class="queue-pill">{obs.get("queue_south", 0)} vehicles</div></div><div class="empty"></div></div></div>'
    return grid_html

def format_analytics_html(history: List[int]) -> str:
    if not history: return "<p style='color:#94a3b8; font-size:0.8em;'>No data yet</p>"
    max_q = max(history) if history else 1
    bars = "".join([f"<div class='spark-bar' style='height: {min((q/30.0)*100, 100)}%; opacity: {0.4+(q/max_q)*0.6};'></div>" for q in history[-30:]])
    return f'<div style="margin-top:10px;"><div class="spark-label">Congestion Trend (Last 30 Steps)</div><div class="sparkline-container">{bars}</div></div>'

def initialize_ui(task: str, seed: int):
    env = SmartAdaptiveTrafficSignalEnv(task_name=task, seed=int(seed))
    obs = env.reset()
    state.update({"env": env, "last_obs": obs.model_dump(), "history": [sum([obs.queue_north, obs.queue_east, obs.queue_south, obs.queue_west])]})
    return format_grid_html(state["last_obs"]), format_analytics_html(state["history"]), f"Ready - Task: {task}", 0, 0.0, f"Score: {env.evaluate():.4f}"

def run_step(phase: str):
    if not state["env"]: return gr.update(), gr.update(), "Reset first", 0, 0.0, "Score: 0.0"
    obs, reward, done, info = state["env"].step(TrafficAction(phase=cast(Phase, phase)))
    state["last_obs"] = obs.model_dump()
    state["history"].append(obs.queue_north + obs.queue_east + obs.queue_south + obs.queue_west)
    return format_grid_html(state["last_obs"]), format_analytics_html(state["history"]), f"Step {obs.step} - {'COMPLETED' if done else 'RUNNING'}", obs.step, reward.value, f"Score: {state['env'].evaluate():.4f}"

def run_auto_move():
    if not state["env"]: return run_step("NS")
    obs = state["last_obs"]
    if obs.get("active_priority"):
        phase = "NS" if obs.get("next_priority_approach") in {"N", "S"} else "EW"
    else:
        phase = "NS" if (obs.get("queue_north", 0) + obs.get("queue_south", 0)) >= (obs.get("queue_east",0) + obs.get("queue_west",0)) else "EW"
    return run_step(phase)

def run_demo():
    with gr.Blocks(title="Smart-Sync Traffic Hub") as demo:
        gr.Markdown("# 🚥 Smart-Sync Adaptive Traffic Hub")
        with gr.Row():
            with gr.Column(scale=2):
                grid_viz = gr.HTML(value="<p style='text-align:center;'>Initialize environment</p>")
                with gr.Accordion("📉 Visual Analytics", open=True):
                    analytics_viz = gr.HTML(value=format_analytics_html([]))
                with gr.Row():
                    phase_input = gr.Radio(["NS", "EW"], value="NS", label="Phase")
                    step_btn = gr.Button("Manual Step", variant="primary")
                    auto_btn = gr.Button("💡 AI Suggestion", variant="secondary")
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Session Config")
                task_input = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Scenario")
                seed_input = gr.Number(42, label="Seed", precision=0)
                reset_btn = gr.Button("Reset Simulation")
                status_txt = gr.Textbox("Not Started", label="Logs", interactive=False)
                gr.HTML("<hr>")
                gr.Markdown("### 📊 Performance Monitor")
                with gr.Row():
                    step_gauge = gr.Number(label="Steps", value=0, interactive=False)
                    reward_gauge = gr.Number(label="Reward", value=0.0, interactive=False)
                score_display = gr.Markdown("## Score: 0.0000")
        with gr.Accordion("Technical Specs & Legend", open=False):
            gr.Markdown("""
            ### Metrics & Reward Structure
            - **NS Phase**: North-South traffic moves.
            - **EW Phase**: East-West traffic moves.
            - **Emergency First**: Agents are penalized heavily for delaying ambulances.
            - **Congestion Trend**: The analytics chart tracks the total vehicle backlog.
            """)
        reset_btn.click(initialize_ui, [task_input, seed_input], [grid_viz, analytics_viz, status_txt, step_gauge, reward_gauge, score_display])
        step_btn.click(run_step, [phase_input], [grid_viz, analytics_viz, status_txt, step_gauge, reward_gauge, score_display])
        auto_btn.click(run_auto_move, None, [grid_viz, analytics_viz, status_txt, step_gauge, reward_gauge, score_display])
    return demo

if __name__ == "__main__":
    run_demo().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS)
