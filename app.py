#!/usr/bin/env python3
"""
Hugging Face Spaces Gradio Interface for Smart Adaptive Traffic Signal OpenEnv
"""

import gradio as gr
from typing import cast
from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction, Phase

TASK_OPTIONS = ["easy", "medium", "hard"]
env_state = {}


def initialize_env(task: str, seed: int) -> dict:
    """Initialize or reset the environment."""
    env = SmartAdaptiveTrafficSignalEnv(task_name=task, seed=seed)
    obs = env.reset()
    env_state[task] = {"env": env, "step": 0}
    return {
        "status": f"Environment initialized: {task}",
        "observation": obs.model_dump(),
    }


def step_env(task: str, phase: str) -> dict:
    """Execute one step in the environment."""
    if task not in env_state:
        return {"error": "Environment not initialized. Call reset first."}

    env = env_state[task]["env"]
    try:
        action = TrafficAction(phase=cast(Phase, phase))
        obs, reward, done, info = env.step(action)
        env_state[task]["step"] += 1
        
        metrics = env.get_metrics()
        score = env.evaluate()
        
        return {
            "step": env_state[task]["step"],
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "metrics": metrics,
            "score": f"{score:.4f}",
            "info": info,
        }
    except Exception as e:
        return {"error": str(e)}


def get_state(task: str) -> dict:
    """Get current environment state."""
    if task not in env_state:
        return {"error": "Environment not initialized."}
    
    env = env_state[task]["env"]
    return {
        "state": env.state(),
        "metrics": env.get_metrics(),
        "score": f"{env.evaluate():.4f}",
    }


def run_demo():
    """Run the Gradio demo."""
    with gr.Blocks(title="Smart Adaptive Traffic Signal") as demo:
        gr.Markdown("# Smart Adaptive Traffic Signal - OpenEnv Demo")
        gr.Markdown(
            "Control a four-way intersection traffic signal. Choose phases (NS or EW) "
            "to minimize queues and prioritize emergency vehicles."
        )

        with gr.Row():
            task_select = gr.Dropdown(
                choices=TASK_OPTIONS,
                value="easy",
                label="Task",
                info="Select task difficulty",
            )
            seed_input = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("Reset Environment")

        reset_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            phase_select = gr.Radio(
                choices=["NS", "EW"],
                value="NS",
                label="Choose Signal Phase",
            )
            step_btn = gr.Button("Step")

        with gr.Row():
            state_btn = gr.Button("Get State")

        step_output = gr.JSON(label="Step Result")
        state_output = gr.JSON(label="State")

        reset_btn.click(
            fn=initialize_env,
            inputs=[task_select, seed_input],
            outputs=reset_output,
        )

        step_btn.click(
            fn=step_env,
            inputs=[task_select, phase_select],
            outputs=step_output,
        )

        state_btn.click(
            fn=get_state,
            inputs=task_select,
            outputs=state_output,
        )

        gr.Markdown(
            """
            ### How to Use
            1. Select a task (easy, medium, or hard)
            2. Click "Reset Environment" to start
            3. Choose a phase (NS or EW) and click "Step" to advance
            4. Click "Get State" to see metrics and score
            
            ### Observation Fields
            - **step**: current step number
            - **phase**: active signal phase
            - **queue_***: number of vehicles queued at each approach
            - **active_priority**: whether an ambulance is waiting
            - **next_priority_approach**: direction of priority vehicle
            - **priority_wait**: how long priority vehicle has waited
            """
        )

    return demo


if __name__ == "__main__":
    demo = run_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
