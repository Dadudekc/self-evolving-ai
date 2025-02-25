import os
import traceback
import subprocess
import time
import shutil
import json
import random
import logging
import importlib
import sys

# Example LLM queries (using Ollama locally)
OLLAMA_MODELS = {
    "debugging": "mistral",
    "optimization": "deepseek-coder",
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------
#  Global Paths & Directories
# -------------------------------------
HISTORY_DIR = "ai_versions"
os.makedirs(HISTORY_DIR, exist_ok=True)

MEMORY_FILE = "ai_memory.json"

# -------------------------------------
#  Memory Helpers
# -------------------------------------
def load_memory():
    """Loads memory from JSON, handling missing or corrupt files."""
    if not os.path.exists(MEMORY_FILE):
        return {"past_performance": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        logging.warning("Memory file corrupted. Resetting AI memory...")
        return {"past_performance": []}

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save memory: {e}")

# -------------------------------------
#  1) SelfImprovingDebugger
# -------------------------------------
class SelfImprovingDebugger:
    """
    Handles:
      - Reading/Analyzing code
      - Querying an AI model for fixes
      - Writing improved code
      - Reloading updated scripts
      - Optionally, auto-deploy (Git)
    """
    def __init__(self, script_path):
        self.script_path = script_path
        self.memory = load_memory()
        self.baseline = 100.0  # Some performance baseline
        self.backup_script()

    def backup_script(self):
        """Creates a backup of the current script before modification."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
            shutil.copy(self.script_path, backup_path)
            logging.info(f"Backup created at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to backup script: {e}")

    def run_script(self):
        """
        Attempts to run the current script to see if it works,
        logs time/performance, and handles errors by self-improving.
        """
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            execution_time = time.time() - start_time
            logging.info(f"Script ran successfully. Execution Time: {execution_time:.4f} sec.")

            # Update memory & baseline with "performance"
            self.memory["past_performance"].append(execution_time)
            save_memory(self.memory)

            # Optionally auto-deploy if performance is better than threshold
            if execution_time < 1.0:
                self.auto_deploy()

            return execution_time
        except Exception:
            logging.error("Error detected during script execution. Initiating evolution process...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        """
        Analyzes errors, queries an AI model to fix/optimize the script,
        tests the improved code, and overwrites if valid.
        """
        current_code = self._read_current_code()
        if current_code is None:
            return self.baseline

        # Step 1: Attempt to fix errors via "debugging" model
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
            logging.error("Debugging failed: No output from model.")
            return self.baseline

        # Step 2: Optimize code via "optimization" model
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
            logging.error("Optimization failed: No output from model.")
            return self.baseline

        # Step 3: Validate new code
        tester = TestDebugger()
        if tester.test_new_code(improved_code):
            # If new code is valid, write and reload it
            try:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                logging.info("Script successfully improved and updated!")
                self.reload_script()
                return self.run_script()
            except Exception as e:
                logging.error(f"Failed to write improved script: {e}")
                return self.baseline
        else:
            logging.warning("AI improvement validation failed. Keeping original script.")
            return self.baseline

    def reload_script(self):
        """Dynamically reload the script so changes take effect immediately."""
        logging.info("Reloading the updated script in-memory...")
        module_name = os.path.splitext(os.path.basename(self.script_path))[0]
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)
            logging.info("✅ Script reloaded successfully!")
        except Exception as e:
            logging.error(f"Failed to reload script: {e}")

    def _read_current_code(self):
        """Utility to read the current script file."""
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read script: {e}")
            return None

    def query_ollama(self, model, prompt):
        """Queries a local LLM (Ollama) for fixes or optimizations."""
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
        """Pushes the script’s latest version to GitHub if any changes are detected."""
        logging.info("Auto-deploying script to GitHub...")
        from git import Repo, InvalidGitRepositoryError

        try:
            try:
                repo = Repo(".", search_parent_directories=True)
            except InvalidGitRepositoryError:
                logging.error("Deployment failed: Not a Git repository.")
                return

            if repo.is_dirty(untracked_files=True):
                repo.git.add(".")
                repo.index.commit("🚀 Auto-commit from SelfImprovingDebugger")
                repo.git.push("origin", "main")
                logging.info("Script successfully deployed to GitHub.")
            else:
                logging.info("No changes detected. Skipping deployment.")
        except Exception as e:
            logging.error(f"Deployment failed: {e}")

# -------------------------------------
#  2) TestDebugger
# -------------------------------------
class TestDebugger:
    """
    Responsible for:
      - Creating isolated test files
      - Executing code
      - Checking for errors
      - Returning pass/fail or success/time
    """
    def __init__(self):
        pass  # Potentially store test results or config if needed

    def test_new_code(self, candidate_code: str) -> bool:
        """
        Writes candidate code to a temp file, tries to exec it.
        Returns True if no errors occur, else False.
        """
        test_path = os.path.join(HISTORY_DIR, "temp_test.py")
        try:
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(candidate_code)

            start_time = time.time()
            with open(test_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            test_time = time.time() - start_time
            logging.info(f"Candidate code test successful (Execution: {test_time:.4f}s)")
            return True
        except Exception as e:
            logging.error(f"Candidate code failed test: {e}")
            return False
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

# -------------------------------------
#  3) ProjectMaker
# -------------------------------------
class ProjectMaker:
    """
    A skeleton class illustrating how you might:
      - Create new Python projects or modules
      - Scaffold directories and files
      - Insert sample code or config
    """
    def __init__(self, base_dir="new_projects"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_new_project(self, project_name: str):
        """Create a new Python project scaffold."""
        project_path = os.path.join(self.base_dir, project_name)
        if os.path.exists(project_path):
            logging.warning(f"Project {project_name} already exists.")
            return project_path

        os.makedirs(project_path)
        logging.info(f"Created new project directory: {project_path}")

        # Create a main script
        main_script_path = os.path.join(project_path, "main.py")
        with open(main_script_path, "w", encoding="utf-8") as f:
            f.write('# Auto-generated main.py\nprint("Hello from new project!")\n')

        # Create a README
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {project_name}\n\nThis is an auto-generated project.\n")

        logging.info("Scaffold created with main.py and README.md.")
        return project_path

# -------------------------------------
#  Example usage
# -------------------------------------
if __name__ == "__main__":
    # 1) We instantiate the SelfImprovingDebugger with the current script
    script_path = __file__
    debugger = SelfImprovingDebugger(script_path)

    try:
        while True:
            # 2) Run the script to see if it errors => triggers improvement
            exec_time = debugger.run_script()
            logging.info(f"Baseline: {debugger.baseline:.2f}, ExecTime: {exec_time:.2f}")

            # 3) Simple logic to lightly adjust baseline
            alpha = 0.1
            debugger.baseline = alpha * exec_time + (1 - alpha) * debugger.baseline

            # 4) Create a new project occasionally (example usage)
            if random.random() < 0.1:
                pm = ProjectMaker()
                new_project_name = f"project_{int(time.time())}"
                pm.create_new_project(new_project_name)

            # Wait
            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("Terminated by user.")
