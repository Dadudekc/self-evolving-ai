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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
        # Initialize with default business metrics
        return {"goals": [], "past_performance": [], "training_data": [], 
                "revenue": 0.0, "engagement": 0.0, "efficiency": 0.0}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
            return memory
    except (json.JSONDecodeError, ValueError):
        logging.warning("Memory file corrupted. Resetting AI memory...")
        return {"goals": [], "past_performance": [], "training_data": [], 
                "revenue": 0.0, "engagement": 0.0, "efficiency": 0.0}

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save memory: {e}")

# Neural Network to Predict Best AI Improvements (input includes 4 features)
class AIImprovementPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=1):
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
        # Business-focused goal options
        self.goal_options = [
            "Find profitable business opportunities",
            "Develop an automated revenue stream",
            "Improve marketing automation",
            "Enhance customer engagement and acquisition",
            "Optimize business operations for efficiency",
            "Automate lead generation",
            "Build and scale a product or service",
            "Develop a sales and monetization strategy",
            "Improve financial tracking and forecasting",
            "Integrate AI-powered customer support",
        ]
        self.backup_script()
        # Baseline for business performance (arbitrary starting value)
        self.baseline = 100.0

    def backup_script(self):
        """Creates a backup of the current script before modification."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
            shutil.copy(self.script_path, backup_path)
            logging.info(f"Backup created at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to backup script: {e}")

    def choose_goal(self):
        """
        Chooses a business goal using ML prediction if sufficient training data exists,
        otherwise selects one randomly.
        Features used: [revenue, engagement, efficiency, goal_index]
        """
        best_goal = None
        best_prediction = -float("inf")
        training_data = self.memory.get("training_data", [])
        if len(training_data) < 5:
            best_goal = random.choice(self.goal_options)
        else:
            revenue = self.memory.get("revenue", 0.0)
            engagement = self.memory.get("engagement", 0.0)
            efficiency = self.memory.get("efficiency", 0.0)
            for idx, goal in enumerate(self.goal_options):
                features = torch.tensor([revenue, engagement, efficiency, idx], dtype=torch.float32)
                prediction = self.model(features)
                if prediction.item() > best_prediction:
                    best_prediction = prediction.item()
                    best_goal = goal
        self.memory["goals"].append(best_goal)
        save_memory(self.memory)
        logging.info(f"AI has chosen goal: {best_goal}")
        return best_goal

    def run_script(self):
        """
        Executes the current script.
        In this business-oriented version, we simulate business performance metrics.
        """
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            execution_time = time.time() - start_time
            logging.info(f"Script ran successfully. Execution Time: {execution_time:.4f} sec.")

            # Simulate business metrics (in a real scenario, these would come from actual business data)
            revenue = random.uniform(50, 500)           # Simulated revenue gain
            engagement = random.uniform(10, 100)          # Simulated customer engagement
            efficiency = random.uniform(5, 50)            # Simulated operational efficiency improvement

            logging.info(f"Revenue: ${revenue:.2f}, Engagement: {engagement:.2f}, Efficiency: {efficiency:.2f}")

            # Save simulated performance metrics
            self.memory["revenue"] = revenue
            self.memory["engagement"] = engagement
            self.memory["efficiency"] = efficiency
            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)
            return execution_time
        except Exception:
            logging.error("Error detected during script execution. Initiating evolution process...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        """Analyzes errors and attempts to improve the script using Ollama."""
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                current_code = f.read()
        except Exception as e:
            logging.error(f"Failed to read current script for improvement: {e}")
            return self.baseline

        logging.info("AI is analyzing and rewriting the script...")

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
        if not fixed_code:
            logging.error("Debugging failed: No output from debugging model.")
            return self.baseline

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
        if not improved_code:
            logging.error("Optimization failed: No output from optimization model.")
            return self.baseline

        # Validate new code
        if self.test_new_code(improved_code):
            try:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                logging.info("Script successfully improved and updated!")
                return self.run_script()
            except Exception as e:
                logging.error(f"Failed to write improved script: {e}")
                return self.baseline
        else:
            logging.warning("AI improvement validation failed. Keeping original script.")
            return self.baseline  # Return baseline as a fallback

    def test_new_code(self, candidate_code):
        """Tests candidate code by executing it. Returns True if it runs without errors."""
        test_path = os.path.join(HISTORY_DIR, "temp_test.py")
        try:
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(candidate_code)
            start_time = time.time()
            with open(test_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            test_time = time.time() - start_time
            logging.info(f"Candidate code ran successfully (Test Execution Time: {test_time:.4f} sec.)")
            return True
        except Exception as e:
            logging.error(f"Candidate code failed during test with error: {e}")
            return False
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def query_ollama(self, model, prompt):
        """Queries Ollama locally for AI-generated fixes or optimizations."""
        try:
            response = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, check=True
            )
            return response.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama query failed for model {model}: {e}")
            return ""

    def auto_deploy(self):
        """Pushes the AI's latest version to GitHub."""
        logging.info("Auto-deploying AI to GitHub...")
        try:
            try:
                repo = Repo(".", search_parent_directories=True)
            except InvalidGitRepositoryError:
                logging.error("Deployment failed: Current directory is not a Git repository.")
                return
            if repo.is_dirty(untracked_files=True):
                repo.git.add(".")
                repo.index.commit("🚀 AI Evolution Update: Auto-commit from AI Agent")
                repo.git.push("origin", "main")
                logging.info("AI successfully deployed to GitHub.")
            else:
                logging.info("No changes detected. Skipping deployment.")
        except Exception as e:
            logging.error(f"Deployment failed: {e}")

    def add_training_sample(self, revenue, engagement, efficiency, goal_idx):
        """
        Creates a training sample with features: [revenue, engagement, efficiency, goal_idx]
        and target as the sum of business improvements.
        """
        improvement = revenue + engagement + efficiency
        sample = {"features": [revenue, engagement, efficiency, goal_idx], "target": improvement}
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
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
        self.model.eval()

    # 4️⃣ AI Business Automation – Sells digital products & reinvests revenue.
    def ai_business_automation(self):
        """Simulates digital product sales and reinvests generated revenue."""
        sale_revenue = random.uniform(100, 1000)
        reinvest_ratio = 0.3
        reinvestment = sale_revenue * reinvest_ratio
        self.memory["revenue"] += sale_revenue + reinvestment
        logging.info(f"Digital product sale: Revenue ${sale_revenue:.2f}, Reinvestment ${reinvestment:.2f}")
        save_memory(self.memory)
        return sale_revenue, reinvestment

    # 5️⃣ AI Social Media Growth – Posts and tracks engagement.
    def ai_social_media_growth(self):
        """Simulates posting to social media and tracking engagement."""
        post_engagement = random.uniform(50, 500)
        self.memory["engagement"] += post_engagement
        logging.info(f"Social media post engagement: {post_engagement:.2f} new engagements")
        save_memory(self.memory)
        return post_engagement

    # 6️⃣ AI Battles – Competes with other AI models in optimization challenges.
    def ai_battle(self):
        """Simulates an AI battle challenge with another AI model."""
        my_score = random.uniform(0, 100)
        opponent_score = random.uniform(0, 100)
        result = "won" if my_score > opponent_score else "lost" if my_score < opponent_score else "tied"
        logging.info(f"AI Battle result: Score {my_score:.2f} vs Opponent {opponent_score:.2f} => {result}")
        return result, my_score, opponent_score

if __name__ == "__main__":
    script = __file__
    agent = SelfLearningAI(script)
    try:
        while True:
            # 1️⃣ Self-Writing AI: Already integrated via code improvement routines.
            # 2️⃣ Memory & Personality AI: Managed via persistent memory functions.
            # 3️⃣ Self-Learning AI: Ongoing model training.
            # Choose a business goal (optionally override manually)
            goal = agent.choose_goal()
            goal_idx = agent.goal_options.index(goal)
            
            # Run the script and measure execution time
            current_exec_time = agent.run_script()
            
            # Record training data using current business metrics
            revenue = agent.memory.get("revenue", 0.0)
            engagement = agent.memory.get("engagement", 0.0)
            efficiency = agent.memory.get("efficiency", 0.0)
            agent.add_training_sample(revenue, engagement, efficiency, goal_idx)
            
            # Train the improvement predictor model with accumulated samples
            agent.train_model(epochs=20)
            
            # Update baseline using exponential moving average for more stability
            alpha = 0.1
            agent.baseline = alpha * revenue + (1 - alpha) * agent.baseline
            logging.info(f"Updated baseline revenue: {agent.baseline:.2f}")
            
            # Optionally deploy if performance criteria are met
            if current_exec_time < 1.0:
                agent.auto_deploy()
            
            # 4️⃣ AI Business Automation: Simulate digital product sales
            if random.random() < 0.5:
                agent.ai_business_automation()
            
            # 5️⃣ AI Social Media Growth: Simulate social media posts
            if random.random() < 0.5:
                agent.ai_social_media_growth()
            
            # 6️⃣ AI Battles: Occasionally engage in an AI battle challenge
            if random.random() < 0.2:
                agent.ai_battle()
            
            # Pause briefly before the next cycle
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("Evolution loop terminated by user.")
