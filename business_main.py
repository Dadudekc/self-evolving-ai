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
import importlib
import sys

# For real business automation (e.g., Gumroad)
import requests

# For real social media growth (e.g., Twitter)
import tweepy

# For AI battles (comparing Hugging Face models)
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------
#  CONFIG / API KEYS (EXAMPLES)
# ------------------------------
# Store these in environment variables or a secure config file.
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")

# Ollama usage (local LLM) – placeholders
OLLAMA_MODELS = {
    "debugging": "mistral",
    "optimization": "deepseek-coder",
}

# Paths
HISTORY_DIR = "ai_versions"
os.makedirs(HISTORY_DIR, exist_ok=True)

MEMORY_FILE = "ai_memory.json"

def load_memory():
    """Loads AI memory, handling missing or corrupt files."""
    if not os.path.exists(MEMORY_FILE):
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

# Neural Network (simple regressor)
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
        self.baseline = 100.0

    # --------------------------
    #  1) Self-Writing AI
    # --------------------------
    def backup_script(self):
        """Creates a backup of the current script before modification."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
            shutil.copy(self.script_path, backup_path)
            logging.info(f"Backup created at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to backup script: {e}")

    def reload_script(self):
        """
        After writing improved code, dynamically reload this script
        so changes take effect without manual restarts.
        """
        logging.info("Reloading the updated script in-memory...")
        module_name = os.path.splitext(os.path.basename(self.script_path))[0]
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)
            logging.info("✅ Script reloaded successfully!")
        except Exception as e:
            logging.error(f"Failed to reload script: {e}")

    def improve_script(self, error_log):
        """Analyzes errors and attempts to improve the script using Ollama (real LLM calls)."""
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                current_code = f.read()
        except Exception as e:
            logging.error(f"Failed to read current script for improvement: {e}")
            return self.baseline

        logging.info("AI is analyzing and rewriting the script...")

        # Step 1: Attempt to fix errors via Mistral
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

        # Step 2: Optimize via DeepSeek
        improved_code = self.query_ollama(
            OLLAMA_MODELS["optimization"],
            f"""Your task: Optimize the following Python script.
Make it more efficient, clean, and maintainable.
Return ONLY the optimized script.

Fixed Script:
{fixed_code}
"""
        )
        if not improved_code:
            logging.error("Optimization failed: No output from optimization model.")
            return self.baseline

        # Validate new code by testing
        if self.test_new_code(improved_code):
            try:
                # Overwrite
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                logging.info("Script successfully improved and updated!")
                # Reload the script in place
                self.reload_script()
                return self.run_script()
            except Exception as e:
                logging.error(f"Failed to write improved script: {e}")
                return self.baseline
        else:
            logging.warning("AI improvement validation failed. Keeping original script.")
            return self.baseline

    def test_new_code(self, candidate_code):
        """Tests the candidate code by executing it locally. Returns True if it runs without errors."""
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

    # --------------------------
    #  2) Memory & Personality
    # --------------------------
    def choose_goal(self):
        """Uses the ML model to pick next objective based on memory/training data."""
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

    # --------------------------
    #  3) Self-Learning AI
    # --------------------------
    def add_training_sample(self, revenue, engagement, efficiency, goal_idx):
        """Creates a training sample with features => [revenue, engagement, efficiency, goal_idx]."""
        improvement = revenue + engagement + efficiency
        sample = {"features": [revenue, engagement, efficiency, goal_idx], "target": improvement}
        self.memory.setdefault("training_data", []).append(sample)
        save_memory(self.memory)
        return sample

    def train_model(self, epochs=20):
        """Trains the improvement predictor model."""
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
                avg_loss = total_loss / len(training_data)
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        self.model.eval()

    # --------------------------
    #  4) AI Business Automation
    # --------------------------
    def get_actual_gumroad_sales(self):
        """
        Pull real sales data from Gumroad's API (requires GUMROAD_ACCESS_TOKEN).
        If token is not provided or request fails, fallback to random simulation.
        """
        if not GUMROAD_ACCESS_TOKEN:
            logging.warning("No Gumroad API token found; using random simulation instead.")
            return random.uniform(50, 500)

        url = "https://api.gumroad.com/v2/sales"
        params = {"access_token": GUMROAD_ACCESS_TOKEN}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            sales_data = response.json()
            sales = sales_data.get("sales", [])
            total_revenue = sum(sale.get("price", 0) for sale in sales)
            logging.info(f"[Gumroad] Total Real Sales: ${total_revenue:.2f}")
            return total_revenue
        except Exception as e:
            logging.error(f"Gumroad API error: {e}")
            return random.uniform(50, 500)

    def ai_business_automation(self):
        """Fetch real or simulated sales data and reinvest revenue."""
        sale_revenue = self.get_actual_gumroad_sales()
        reinvest_ratio = 0.3
        reinvestment = sale_revenue * reinvest_ratio
        self.memory["revenue"] += sale_revenue + reinvestment

        logging.info(f"Digital product sale: Revenue ${sale_revenue:.2f}, Reinvestment ${reinvestment:.2f}")
        save_memory(self.memory)
        return sale_revenue, reinvestment

    # --------------------------
    #  5) AI Social Media Growth
    # --------------------------
    def post_to_twitter(self, content="Hello from AI Agent!"):
        """
        Post real content to Twitter (X) using your developer credentials.
        If credentials are missing, fallback to random simulation.
        """
        if not (TWITTER_API_KEY and TWITTER_API_SECRET and TWITTER_ACCESS_TOKEN and TWITTER_ACCESS_SECRET):
            logging.warning("Missing Twitter API credentials; simulating post & engagement.")
            return random.uniform(10, 100)

        try:
            auth = tweepy.OAuth1UserHandler(
                TWITTER_API_KEY, TWITTER_API_SECRET,
                TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
            )
            api = tweepy.API(auth)
            status = api.update_status(content)
            logging.info(f"✅ Tweet posted: {status.text}")
            
            # (Optional) get tweet stats
            # stats = api.get_status(status.id, tweet_mode="extended")
            # engagement = stats.favorite_count + stats.retweet_count
            engagement = random.uniform(10, 100)  # Real engagement check requires more advanced usage
            return engagement
        except Exception as e:
            logging.error(f"Failed to post to Twitter: {e}")
            return random.uniform(10, 100)

    def ai_social_media_growth(self):
        """Posts a message to Twitter and updates engagement based on real or simulated response."""
        content = "🚀 AI Social Media Growth in Action!"
        engagement_gain = self.post_to_twitter(content)
        self.memory["engagement"] += engagement_gain
        logging.info(f"Social media post engagement: {engagement_gain:.2f} new engagements")
        save_memory(self.memory)
        return engagement_gain

    # --------------------------
    #  6) AI Battles
    # --------------------------
    def ai_battle(self):
        """
        Compare your model's output against a Hugging Face model as a "battle."
        Here we do a basic text classification for demonstration.
        """
        try:
            # Your model (placeholder) – you could load a local model
            # For demonstration, we use a built-in pipeline (like "distilbert-base-uncased-finetuned-sst-2-english")
            local_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

            # Opponent: Another HF model
            opponent_pipeline = pipeline("text-classification", model="bert-base-uncased")

            text = "AI competition is a great way to advance automation!"
            my_result = local_pipeline(text)[0]
            opp_result = opponent_pipeline(text)[0]

            # We'll "score" positivity
            my_score = my_result["score"] if my_result["label"] == "POSITIVE" else 1 - my_result["score"]
            opp_score = opp_result["score"] if opp_result["label"] == "POSITIVE" else 1 - opp_result["score"]

            if my_score > opp_score:
                result = "won"
            elif my_score < opp_score:
                result = "lost"
            else:
                result = "tied"

            logging.info(f"AI Battle result: My Score={my_score:.2f} vs Opponent={opp_score:.2f} => {result}")
            return result, my_score, opp_score

        except Exception as e:
            logging.error(f"AI Battle error: {e}")
            # fallback: random
            my_score = random.uniform(0,100)
            opponent_score = random.uniform(0,100)
            result = "won" if my_score>opponent_score else "lost" if my_score<opponent_score else "tied"
            return result, my_score, opponent_score

    # --------------------------
    #  Script Workflow
    # --------------------------
    def run_script(self):
        """Main business logic, previously simulating but now partially real if APIs are configured."""
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            execution_time = time.time() - start_time
            logging.info(f"Script ran successfully. Execution Time: {execution_time:.4f} sec.")

            # Real or random revenue (see get_actual_gumroad_sales as an example).
            # We'll do a fallback random for "extra" revenue besides the dedicated business_automation method.
            revenue = random.uniform(50, 500)
            engagement = random.uniform(10, 100)
            efficiency = random.uniform(5, 50)

            self.memory["revenue"] += revenue
            self.memory["engagement"] += engagement
            self.memory["efficiency"] += efficiency

            logging.info(f"[Simulation] +${revenue:.2f} revenue, +{engagement:.2f} engagement, +{efficiency:.2f} efficiency")

            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)
            return execution_time
        except Exception:
            logging.error("Error detected during script execution. Initiating evolution process...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

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

if __name__ == "__main__":
    script = __file__
    agent = SelfLearningAI(script)
    try:
        while True:
            # Choose / update goal
            goal = agent.choose_goal()
            goal_idx = agent.goal_options.index(goal)

            # Run the script
            current_exec_time = agent.run_script()

            # Record training data
            revenue = agent.memory.get("revenue", 0.0)
            engagement = agent.memory.get("engagement", 0.0)
            efficiency = agent.memory.get("efficiency", 0.0)
            agent.add_training_sample(revenue, engagement, efficiency, goal_idx)

            # Train the AI model
            agent.train_model(epochs=20)

            # Update baseline w/ EMA
            alpha = 0.1
            agent.baseline = alpha * revenue + (1 - alpha) * agent.baseline
            logging.info(f"Updated baseline revenue: {agent.baseline:.2f}")

            # Auto-deploy if performance is quick
            if current_exec_time < 1.0:
                agent.auto_deploy()

            # Real or random business automation
            if random.random() < 0.3:
                agent.ai_business_automation()

            # Post to Twitter or simulate
            if random.random() < 0.3:
                agent.ai_social_media_growth()

            # AI battle vs HF pipeline
            if random.random() < 0.2:
                agent.ai_battle()

            # Wait
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("Evolution loop terminated by user.")
