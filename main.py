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

# AI Memory, Goal Tracking & Training Data
MEMORY_FILE = "ai_memory.json"

def load_memory():
    """Loads AI memory, handling missing or corrupt files."""
    if not os.path.exists(MEMORY_FILE):
        return {"goals": [], "past_performance": [], "training_data": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
            return memory
    except (json.JSONDecodeError, ValueError):
        print("\n⚠️ Memory file corrupted. Resetting AI memory...\n")
        return {"goals": [], "past_performance": [], "training_data": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

# Neural Network to Predict Best AI Improvements
class AIImprovementPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super(AIImprovementPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SelfLearningAI:
    def __init__(self, script_path):
        self.script_path = script_path
        self.memory = load_memory()
        self.model = AIImprovementPredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.goal_options = [
            "Reduce execution time",
            "Improve memory efficiency",
            "Enhance error handling",
            "Add new features",
            "Experiment with new logic",
        ]
        self.backup_script()
        # Baseline execution time (set high initially)
        self.baseline = 5.0

    def backup_script(self):
        """Creates a backup of the current script before modification."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
        shutil.copy(self.script_path, backup_path)

    def choose_goal(self):
        """
        Chooses a goal either by ML prediction or manually.
        The model is fed features: [baseline, last execution time, goal_index].
        For each possible goal, it predicts an expected improvement.
        You can override this logic to direct AI growth.
        """
        best_goal = None
        best_prediction = -float("inf")
        # If not enough training data, choose randomly
        if len(self.memory.get("training_data", [])) < 5:
            best_goal = random.choice(self.goal_options)
        else:
            # Try each goal option and pick the one with best predicted improvement
            for idx, goal in enumerate(self.goal_options):
                # Build a dummy feature vector: [baseline, current baseline, goal_index]
                # (In practice, you might use more meaningful features)
                features = torch.tensor([self.baseline, self.baseline, idx], dtype=torch.float32)
                prediction = self.model(features)
                if prediction.item() > best_prediction:
                    best_prediction = prediction.item()
                    best_goal = goal
        self.memory["goals"].append(best_goal)
        save_memory(self.memory)
        print(f"\n🎯 AI has chosen goal: {best_goal}")
        return best_goal

    def run_script(self):
        """Executes the current script and benchmarks performance."""
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            execution_time = time.time() - start_time
            print(f"\n✅ Script ran successfully. Execution Time: {execution_time:.4f} sec.")
            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)
            return execution_time
        except Exception:
            print("\n❌ Error detected! AI is evolving...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        """Analyzes errors and attempts to improve the script using Ollama."""
        with open(self.script_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        print("\n🤖 AI is analyzing and rewriting the script...")

        # Step 1: Use Mistral (debugging) to fix the error
        fixed_code = self.query_ollama(
            OLLAMA_MODELS["debugging"],
            f"""Your task: Fix the following Python script based on this error log.
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
            f"""Your task: Optimize the following Python script.
Make the code more efficient, clean, and maintainable.
Return ONLY the optimized script.

Fixed Script:
{fixed_code}
"""
        )

        # Validate new code
        if self.test_new_code(improved_code):
            with open(self.script_path, "w", encoding="utf-8") as f:
                f.write(improved_code)
            print("\n🚀 Script successfully improved!")
            return self.run_script()
        else:
            print("\n⚠️ AI improvement failed. Keeping original script.")
            return self.baseline  # Return baseline as a fallback

    def test_new_code(self, candidate_code):
        """Tests candidate code by executing it. Returns True if it runs without errors."""
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
            repo = None
            try:
                repo = Repo(".", search_parent_directories=True)
            except InvalidGitRepositoryError:
                print("\n❌ Deployment failed: Current directory is not a Git repository.")
                return
            if repo.is_dirty(untracked_files=True):
                repo.git.add(".")
                repo.index.commit("🚀 AI Evolution Update: Auto-commit from AI Agent")
                repo.git.push("origin", "main")
                print("\n✅ AI successfully deployed to GitHub.\n")
            else:
                print("\n⚠️ No changes detected. Skipping deployment.")
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}\n")

    def add_training_sample(self, baseline, exec_time, goal_idx):
        """
        Creates a training sample with features: [baseline, exec_time, goal_idx]
        and target as improvement (baseline - exec_time, if positive).
        """
        improvement = max(baseline - exec_time, 0)
        sample = {"features": [baseline, exec_time, goal_idx], "target": improvement}
        self.memory.setdefault("training_data", []).append(sample)
        save_memory(self.memory)
        return sample

    def train_model(self, epochs=20):
        """Trains the improvement predictor using collected training samples."""
        training_data = self.memory.get("training_data", [])
        if not training_data:
            return
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for sample in training_data:
                features = torch.tensor(sample["features"], dtype=torch.float32)
                target = torch.tensor([sample["target"]], dtype=torch.float32)
                self.optimizer.zero_grad()
                output = self.model(features)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # Optionally print loss every few epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
        self.model.eval()

if __name__ == "__main__":
    script = __file__
    agent = SelfLearningAI(script)
    
    # Endless evolution loop
    while True:
        # Choose a goal (optionally, you can override this choice manually)
        goal = agent.choose_goal()
        goal_idx = agent.goal_options.index(goal)
        
        # Run the script and measure execution time
        current_exec_time = agent.run_script()
        
        # Record training data using baseline vs. current performance
        sample = agent.add_training_sample(agent.baseline, current_exec_time, goal_idx)
        
        # Train the improvement predictor model with accumulated samples
        agent.train_model(epochs=20)
        
        # Update baseline (simple moving average with new sample)
        agent.baseline = (agent.baseline + current_exec_time) / 2.0
        print(f"\n📊 Updated baseline execution time: {agent.baseline:.4f} sec.\n")
        
        # Optionally deploy if performance criteria are met
        if current_exec_time < 1.0:
            agent.auto_deploy()
        
        # Pause briefly before the next cycle
        time.sleep(2)
