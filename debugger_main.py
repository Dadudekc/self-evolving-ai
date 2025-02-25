import os
import sys
import time
import json
import random
import shutil
import logging
import traceback
import subprocess
import importlib

# ------------------------------
# Global Constants & Setup
# ------------------------------
CONTEXT_MEMORY_FILE = "context_memory.json"  # For project plans & contextual memory
DEBUG_MEMORY_FILE = "ai_memory.json"         # For debugger performance memory
HISTORY_DIR = "ai_versions"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Example LLM query integrations (using Ollama locally)
OLLAMA_MODELS = {
    "debugging": "mistral",
    "optimization": "deepseek-coder",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------
# 1️⃣ MemoryManager (Contextual Memory)
# ------------------------------
class MemoryManager:
    """Stores long-term project plans, unfinished & completed steps."""
    def __init__(self):
        self.memory = self.load_memory()

    def load_memory(self):
        if not os.path.exists(CONTEXT_MEMORY_FILE):
            return {"project_plan": None, "unfinished_steps": [], "completed_steps": []}
        try:
            with open(CONTEXT_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            logging.warning("Context memory file corrupted. Resetting memory...")
            return {"project_plan": None, "unfinished_steps": [], "completed_steps": []}

    def save_memory(self):
        try:
            with open(CONTEXT_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save context memory: {e}")

    def set_plan(self, plan):
        self.memory["project_plan"] = plan
        self.memory["unfinished_steps"] = plan.copy()
        self.memory["completed_steps"] = []
        self.save_memory()

    def get_next_step(self):
        if self.memory["unfinished_steps"]:
            return self.memory["unfinished_steps"][0]
        return None

    def mark_step_complete(self, step):
        if step in self.memory["unfinished_steps"]:
            self.memory["unfinished_steps"].remove(step)
            self.memory["completed_steps"].append(step)
            self.save_memory()

# ------------------------------
# 2️⃣ ProjectPlanner (Plan Generator)
# ------------------------------
class ProjectPlanner:
    """Generates a structured step-by-step roadmap for a new project."""
    def __init__(self, project_name):
        self.project_name = project_name

    def generate_plan(self):
        plan = [
            f"Create project directory for {self.project_name}",
            f"Generate main.py with initial structure",
            f"Create README.md",
            f"Set up virtual environment",
            f"Install dependencies",
            f"Write unit tests",
            f"Implement core functionality",
            f"Debug and optimize code",
            f"Run final validation tests",
            f"Deploy the project",
        ]
        return plan

# ------------------------------
# 3️⃣ SelfImprovingDebugger (Debug/Test & Auto-Improve)
# ------------------------------
class SelfImprovingDebugger:
    """
    Handles code improvement:
      - Backs up current script.
      - Attempts to run the script.
      - On error, queries an LLM (via Ollama) for fixes/optimizations.
      - Tests the improved code and reloads if valid.
      - Optionally auto-deploys via Git.
    """
    def __init__(self, script_path):
        self.script_path = script_path
        self.memory = self.load_debug_memory()
        self.baseline = 100.0
        self.backup_script()

    def load_debug_memory(self):
        if not os.path.exists(DEBUG_MEMORY_FILE):
            return {"past_performance": []}
        try:
            with open(DEBUG_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            logging.warning("Debug memory file corrupted. Resetting...")
            return {"past_performance": []}

    def save_debug_memory(self):
        try:
            with open(DEBUG_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save debug memory: {e}")

    def backup_script(self):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = os.path.join(HISTORY_DIR, f"version_{timestamp}.py")
            shutil.copy(self.script_path, backup_path)
            logging.info(f"Backup created at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to backup script: {e}")

    def run_script(self):
        start_time = time.time()
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            exec_time = time.time() - start_time
            logging.info(f"Script ran successfully in {exec_time:.4f} sec.")

            self.memory["past_performance"].append(exec_time)
            self.save_debug_memory()

            if exec_time < 1.0:
                self.auto_deploy()
            return exec_time
        except Exception:
            logging.error("Error during script execution. Initiating improvement...")
            error_details = traceback.format_exc()
            return self.improve_script(error_details)

    def improve_script(self, error_log):
        current_code = self._read_current_code()
        if current_code is None:
            return self.baseline

        fixed_code = self.query_ollama(
            OLLAMA_MODELS["debugging"],
            f"""Fix the following script based on the error log:
Error Log:
{error_log}
Current Script:
{current_code}
Return ONLY the corrected script."""
        )
        if not fixed_code:
            logging.error("Debugging model returned no output.")
            return self.baseline

        improved_code = self.query_ollama(
            OLLAMA_MODELS["optimization"],
            f"""Optimize the following fixed script for clarity and efficiency:
Fixed Script:
{fixed_code}
Return ONLY the optimized script."""
        )
        if not improved_code:
            logging.error("Optimization model returned no output.")
            return self.baseline

        tester = TestDebugger()
        if tester.test_new_code(improved_code):
            try:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                logging.info("Script improved and updated successfully!")
                self.reload_script()
                return self.run_script()
            except Exception as e:
                logging.error(f"Error writing improved script: {e}")
                return self.baseline
        else:
            logging.warning("New code failed tests. Keeping original script.")
            return self.baseline

    def reload_script(self):
        logging.info("Reloading updated script...")
        module_name = os.path.splitext(os.path.basename(self.script_path))[0]
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)
            logging.info("Script reloaded successfully!")
        except Exception as e:
            logging.error(f"Failed to reload script: {e}")

    def _read_current_code(self):
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read script: {e}")
            return None

    def query_ollama(self, model, prompt):
        try:
            response = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, check=True
            )
            return response.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama query failed for {model}: {e}")
            return ""

    def auto_deploy(self):
        logging.info("Auto-deploying script to GitHub...")
        from git import Repo, InvalidGitRepositoryError
        try:
            repo = Repo(".", search_parent_directories=True)
        except InvalidGitRepositoryError:
            logging.error("Not a Git repository. Skipping deployment.")
            return
        if repo.is_dirty(untracked_files=True):
            repo.git.add(".")
            repo.index.commit("🚀 Auto-commit from SelfImprovingDebugger")
            repo.git.push("origin", "main")
            logging.info("Script deployed to GitHub.")
        else:
            logging.info("No changes to deploy.")

# ------------------------------
# 4️⃣ TestDebugger (Unit Test Runner)
# ------------------------------
class TestDebugger:
    """Executes tests to validate functionality."""
    def run_tests(self):
        logging.info("Running automated tests...")
        # Placeholder: In a real implementation, integrate with pytest or similar.
        return True

    def test_new_code(self, candidate_code: str) -> bool:
        test_path = os.path.join(HISTORY_DIR, "temp_test.py")
        try:
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(candidate_code)
            start_time = time.time()
            with open(test_path, "r", encoding="utf-8") as f:
                exec(f.read(), {})
            elapsed = time.time() - start_time
            logging.info(f"Candidate code executed in {elapsed:.4f}s without errors.")
            return True
        except Exception as e:
            logging.error(f"Candidate code failed test: {e}")
            return False
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

# ------------------------------
# 5️⃣ ProjectExecutor (Plan Follower)
# ------------------------------
class ProjectExecutor:
    """
    Executes the project plan step by step:
      - Retrieves next step via MemoryManager.
      - Calls appropriate actions (create directory, generate files, run tests, debug code).
    """
    def __init__(self, memory_manager, debugger, test_runner):
        self.memory = memory_manager
        self.debugger = debugger
        self.test_runner = test_runner

    def execute_next_step(self):
        step = self.memory.get_next_step()
        if not step:
            logging.info("✅ All project steps are complete!")
            return
        logging.info(f"📌 Executing step: {step}")
        success = self.perform_action(step)
        if success:
            self.memory.mark_step_complete(step)
            logging.info(f"✅ Step completed: {step}")
        else:
            logging.warning(f"❌ Step failed: {step}. Retrying...")

    def perform_action(self, step):
        if "Create project directory" in step:
            return self.create_project_directory()
        elif "Generate main.py" in step:
            return self.generate_main_script()
        elif "Create README.md" in step:
            return self.create_readme()
        elif "Write unit tests" in step:
            return self.test_runner.run_tests()
        elif "Debug and optimize code" in step:
            # Here we expect the debugger to run quickly if code is good.
            return self.debugger.run_script() < 5.0
        else:
            logging.info(f"Simulated execution of: {step}")
            return True

    def create_project_directory(self):
        project_name = self.memory.memory["project_plan"][0] if self.memory.memory["project_plan"] else "default_project"
        path = os.path.join("projects", project_name)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created project directory: {path}")
            return True
        logging.warning(f"Project directory already exists: {path}")
        return False

    def generate_main_script(self):
        project_name = self.memory.memory["project_plan"][0] if self.memory.memory["project_plan"] else "default_project"
        path = os.path.join("projects", project_name, "main.py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Auto-generated main.py\nprint('Hello World')\n")
            logging.info(f"Generated main.py at {path}")
            return True
        logging.warning(f"main.py already exists at {path}")
        return False

    def create_readme(self):
        project_name = self.memory.memory["project_plan"][0] if self.memory.memory["project_plan"] else "default_project"
        path = os.path.join("projects", project_name, "README.md")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {project_name}\n\nThis is an auto-generated project.")
            logging.info(f"Generated README.md at {path}")
            return True
        logging.warning(f"README.md already exists at {path}")
        return False

# ------------------------------
# (Optional) 6️⃣ ProjectMaker (Standalone Scaffold Creator)
# ------------------------------
class ProjectMaker:
    """
    Illustrates scaffolding a new project independently.
    Can be invoked separately from plan execution.
    """
    def __init__(self, base_dir="new_projects"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_new_project(self, project_name: str):
        project_path = os.path.join(self.base_dir, project_name)
        if os.path.exists(project_path):
            logging.warning(f"Project {project_name} already exists.")
            return project_path
        os.makedirs(project_path)
        logging.info(f"Created project directory: {project_path}")
        main_script_path = os.path.join(project_path, "main.py")
        with open(main_script_path, "w", encoding="utf-8") as f:
            f.write('# Auto-generated main.py\nprint("Hello from new project!")\n')
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {project_name}\n\nThis is an auto-generated project.\n")
        logging.info("Scaffold created with main.py and README.md.")
        return project_path

# ------------------------------
# Main Execution Flow
# ------------------------------
if __name__ == "__main__":
    # 1️⃣ Initialize contextual memory for project planning.
    context_memory = MemoryManager()
    if not context_memory.memory["project_plan"]:
        logging.info("No active project plan found. Generating a new plan...")
        project_name = f"project_{int(time.time())}"
        planner = ProjectPlanner(project_name)
        plan = planner.generate_plan()
        context_memory.set_plan(plan)
        logging.info(f"📜 New project plan: {plan}")

    # 2️⃣ Instantiate core components.
    project_name = context_memory.memory["project_plan"][0] if context_memory.memory["project_plan"] else "default_project"
    project_dir = os.path.join("projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    script_path = os.path.join(project_dir, "main.py")
    if not os.path.exists(script_path):
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated placeholder\nprint('Hello World')\n")
    debugger = SelfImprovingDebugger(script_path)
    tester = TestDebugger()
    executor = ProjectExecutor(context_memory, debugger, tester)

    # 3️⃣ Execute the project plan step by step.
    while context_memory.get_next_step():
        executor.execute_next_step()
        time.sleep(2)  # Wait between steps

    logging.info("✅ Project fully completed!")
