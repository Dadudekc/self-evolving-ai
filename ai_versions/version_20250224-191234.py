import os
import traceback
import subprocess
import time
import shutil
import difflib
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from git import Repo  # Requires `pip install gitpython`
from collections import deque  # For tracking improvements over time

# Use Ollama with Mistral & DeepSeek for AI competition
OLLAMA_MODELS = {
    "debugging": "mistral",
    "optimization": "deepseek-coder",
}

# AI Version History
HISTORY_DIR = "ai_versions"
if not os.path.exists(HISTORY_DIR):
    os.mkdir(HISTORY_DIR)

# AI Memory & Goal Tracking
MEMORY_FILE = "ai_memory.json"

# GitHub Repo for Auto-Deployment
GITHUB_REPO = "D:/side_projects/self-evolving-ai"


# Neural Network to Predict Best AI Improvements
class AIImprovementPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super(AIImprovementPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def load_memory():
    """Load AI memory, handling empty or corrupt files."""
    if not os.path.exists(MEMORY_FILE):  # If file doesn't exist, create default memory
        return {"goals": [], "past_performance": deque(maxlen=10)}
    
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
            memory["past_performance"] = deque(memory.get("past_performance", []), maxlen=10)  # Convert list back to deque
            return memory
    except (json.JSONDecodeError, ValueError):  # Handle corrupt/empty JSON
        print("\n⚠️ Memory file corrupted. Resetting AI memory...\n")
        return {"goals": [], "past_performance": deque(maxlen=10)}

def save_memory(memory):
    """Convert deque to list before saving to JSON."""
    memory["past_performance"] = list(memory["past_performance"])  # Convert deque to list
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

class SelfLearningAI:
    def __init__(self, script_path):
        self.script_path = script_path
        self.memory = load_memory()
        self.model = AIImprovementPredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.backup_script()

    def backup_script(self):
        """Saves the current script version before modification."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
        shutil.copy(self.script_path, backup_path)

    def set_ai_goal(self):
        """AI dynamically sets its learning objectives."""
        goal_options = [
            "Reduce execution time",
            "Improve memory efficiency",
            "Enhance error handling",
            "Add new features",
            "Experiment with new logic"
        ]
        goal = random.choice(goal_options)
        self.memory["goals"].append(goal)
        save_memory(self.memory)
        print(f"\n🎯 AI has set a new goal: {goal}")

    def run_script(self):
        """Executes the current script and benchmarks performance."""
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:  # ✅ Fix: Force UTF-8
                exec(f.read(), {})
            execution_time = time.time() - start_time
            print(f"\n✅ No errors. Execution Time: {execution_time:.4f} sec.")
            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)
            return execution_time
        except Exception as e:
            print("\n❌ Error detected! AI is evolving...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        """AI competition mode: Multiple versions compete, the best one survives."""
        with open(self.script_path, "r", encoding="utf-8") as f:  # ✅ Fix: Force UTF-8
            current_code = f.read()

        print("\n🤖 AI is analyzing and rewriting the script...")


    def query_ollama(self, model, prompt):
        """Queries Ollama locally to process AI tasks."""
        response = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )
        return response.stdout.strip()

    def mutate_code(self, code):
        """Randomly mutates the AI-generated code to encourage evolution."""
        mutations = [
            lambda c: c.replace("import", "# import"),  
            lambda c: c.replace("def ", "async def "),  
            lambda c: c.replace("try:", "try:\n    # AI Mutation: Testing new logic\n"),  
            lambda c: c + "\n# AI Mutation: Added experimental logic\n"
        ]
        
        if random.random() < 0.3:  
            mutation = random.choice(mutations)
            mutated_code = mutation(code)
            print("\n🧬 AI Mutation Applied!\n")
            return mutated_code
        return code  

    def select_best_version(self, versions):
        """Runs each AI-generated version, benchmarks performance, and selects the best one."""
        execution_times = {}
        for i, version in enumerate(versions):
            temp_script = os.path.join(HISTORY_DIR, f"temp_version_{i}.py")
            with open(temp_script, "w") as f:
                f.write(version)

            try:
                start_time = time.time()
                exec(version, {})
                execution_times[i] = time.time() - start_time
                print(f"✅ Version {i} executed in {execution_times[i]:.4f} sec.")
            except Exception as e:
                print(f"❌ Version {i} failed. Skipping.")

        if execution_times:
            best_version_idx = min(execution_times, key=execution_times.get)  
            print(f"\n🏆 Best AI Version: {best_version_idx} (Fastest Execution Time)\n")
            return versions[best_version_idx]
        return None

    def auto_deploy(self):
        """Pushes the AI's latest version to GitHub."""
        print("\n🚀 Auto-deploying AI to GitHub...\n")
        try:
            repo = Repo("D:/side_projects/self-evolving-ai")  # Use local repo path
            repo.git.add(".")
            repo.index.commit("🚀 AI Evolution Update: Auto-commit from AI Agent")
            repo.git.push("origin", "main")
            print("\n✅ AI successfully deployed to GitHub.\n")
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}\n")

# Run the AI Agent
if __name__ == "__main__":
    script = __file__
    agent = SelfLearningAI(script)
    agent.set_ai_goal()
    execution_time = agent.run_script()

    if execution_time and execution_time < 1.0:
        agent.auto_deploy()
