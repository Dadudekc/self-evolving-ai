import os
import traceback
import subprocess
import time
import shutil
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from git import Repo, InvalidGitRepositoryError
from collections import deque

# Use Ollama with Mistral & DeepSeek for AI competition
OLLAMA_MODELS = {
    "debugging": "mistral",
    "optimization": "deepseek-coder",
}

# AI Version History
HISTORY_DIR = "ai_versions"
os.makedirs(HISTORY_DIR, exist_ok=True)

# AI Memory & Goal Tracking
MEMORY_FILE = "ai_memory.json"

# GitHub Repo for Auto-Deployment (HTTPS URL to your GitHub repo)
GITHUB_REPO = "https://github.com/Dadudekc/self-evolving-ai"

# Neural Network to Predict Best AI Improvements
class AIImprovementPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super(AIImprovementPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def load_memory():
    """Loads AI memory, handling missing or corrupt files."""
    if not os.path.exists(MEMORY_FILE):
        return {"goals": [], "past_performance": deque(maxlen=10)}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
            memory["past_performance"] = deque(memory.get("past_performance", []), maxlen=10)
            return memory
    except (json.JSONDecodeError, ValueError):
        print("\n⚠️ Memory file corrupted. Resetting AI memory...\n")
        return {"goals": [], "past_performance": deque(maxlen=10)}

def save_memory(memory):
    """Converts deque to list before saving to JSON."""
    memory["past_performance"] = list(memory["past_performance"])
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

class SelfLearningAI:
    def __init__(self, script_path):
        self.script_path = script_path
        self.memory = load_memory()
        self.model = AIImprovementPredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.backup_script()

    def backup_script(self):
        """Creates a backup of the current script before modification."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
        shutil.copy(self.script_path, backup_path)

    def set_ai_goal(self):
        """AI dynamically sets its own learning objectives."""
        goal_options = [
            "Reduce execution time",
            "Improve memory efficiency",
            "Enhance error handling",
            "Add new features",
            "Experiment with new logic",
        ]
        goal = random.choice(goal_options)
        self.memory["goals"].append(goal)
        save_memory(self.memory)
        print(f"\n🎯 AI has set a new goal: {goal}")

    def run_script(self):
        """Executes the current script and benchmarks performance."""
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            execution_time = time.time() - start_time
            print(f"\n✅ No errors. Execution Time: {execution_time:.4f} sec.")
            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)
            return execution_time
        except Exception:
            print("\n❌ Error detected! AI is evolving...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        """AI analyzes errors and attempts to improve the script using Mistral & DeepSeek."""
        with open(self.script_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        print("\n🤖 AI is analyzing and rewriting the script...")

        # Step 1: Use Mistral (debugging) to fix the error
        fixed_code = self.query_ollama(
            OLLAMA_MODELS["debugging"],
            f"""\
Your task: Fix the following Python script based on this error log. 
Ensure the script can run without errors. Return ONLY the corrected script.

Error Log:
{error_log}

Current Script:
{current_code}
"""
        )

        # Step 2: Use DeepSeek (optimization) to improve efficiency
        improved_code = self.query_ollama(
            OLLAMA_MODELS["optimization"],
            f"""\
Your task: Optimize the following Python script. 
Make the code more efficient, clean, and maintainable. 
Return ONLY the optimized script.

Fixed Script:
{fixed_code}
"""
        )

        # Step 3: Validate new code
        if self.test_new_code(improved_code):
            # Step 4: Overwrite original script if new code passes
            with open(self.script_path, "w", encoding="utf-8") as f:
                f.write(improved_code)
            print("\n🚀 Script successfully improved! Restarting...\n")
            return self.run_script()
        else:
            print("\n⚠️ AI improvement failed. Keeping original script.")
            return None

    def test_new_code(self, candidate_code):
        """Tests the candidate code by executing it. Returns True if it runs without errors."""
        test_path = os.path.join(HISTORY_DIR, "temp_test.py")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)

        try:
            start_time = time.time()
            with open(test_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            test_time = time.time() - start_time
            print(f"✅ Candidate code ran successfully (Test Execution Time: {test_time:.4f} sec.)")
            return True
        except Exception as e:
            print(f"❌ Candidate code failed with error:\n{e}")
            return False

    def query_ollama(self, model, prompt):
        """Queries Ollama locally for AI-generated fixes or optimizations."""
        response = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )
        return response.stdout.strip()

    def auto_deploy(self):
        """Pushes the AI's latest version to GitHub."""
        print("\n🚀 Auto-deploying AI to GitHub...\n")
        try:
            # Attempt to open current directory as a Git repo (assuming code is in a cloned repo)
            repo = None
            try:
                repo = Repo(".", search_parent_directories=True)
            except InvalidGitRepositoryError:
                print("\n❌ Deployment failed: Current directory is not a Git repository.")
                return

            # Stage, commit, and push changes
            if repo.is_dirty(untracked_files=True):
                repo.git.add(".")
                repo.index.commit("🚀 AI Evolution Update: Auto-commit from AI Agent")
                repo.git.push("origin", "main")
                print("\n✅ AI successfully deployed to GitHub.\n")
            else:
                print("\n⚠️ No changes detected. Skipping deployment.")
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}\n")

# Run the AI Agent
if __name__ == "__main__":
    script = __file__
    agent = SelfLearningAI(script)
    agent.set_ai_goal()
    execution_time = agent.run_script()

    # Example: Deploy automatically if script runs under 1 second
    if execution_time and execution_time < 1.0:
        agent.auto_deploy()
