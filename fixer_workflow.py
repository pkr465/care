import os
import sys
import argparse
from pathlib import Path

# Import helper classes
try:
    from agents.parsers.excel_to_agent_parser import ExcelToAgentParser
    from agents.codebase_fixer_agent import CodebaseFixerAgent
except ImportError:
    print("[!] Error: Could not import helper classes.")
    print("    Ensure 'agents' package is in PYTHONPATH or current directory.")
    sys.exit(1)

class HumanInTheLoopWorkflow:
    """
    Orchestrator for the QGenie Repair Pipeline.
    
    Bridge between:
    1. Human Review (Excel) -> 2. Parsing Logic -> 3. Execution Agent
    """

    def __init__(self, args):
        """
        Initialize the workflow with parsed CLI arguments.
        """
        self.args = args
        self.excel_path = Path(args.excel_file).resolve()
        self.codebase_root = Path(args.codebase_path).resolve()
        
        # FIX: CLI arg is '--out-dir', so we access 'args.out_dir' (not args.workspace)
        self.workspace_dir = Path(args.out_dir).resolve()
        
        # Define artifacts paths
        self.directives_jsonl = self.workspace_dir / "agent_directives.jsonl"
        self.final_report = self.workspace_dir / "final_execution_audit.xlsx"

        # Ensure workspace exists
        os.makedirs(self.workspace_dir, exist_ok=True)

    def execute(self):
        print("="*60)
        print(" QGenie Human-in-the-Loop Repair Workflow")
        print("="*60)

        # Validate inputs before starting
        if not self.codebase_root.exists():
            print(f"[!] Error: Codebase path does not exist: {self.codebase_root}")
            return
        if not self.excel_path.exists():
            print(f"[!] Error: Excel file does not exist: {self.excel_path}")
            return

        # Step 1: Parse the Excel File
        if not self._step_parse_excel():
            print("[!] Workflow aborted at Step 1.")
            return

        # Step 2: Run the Fixer Agent
        self._step_run_agent()

        print("\n" + "="*60)
        print(" WORKFLOW COMPLETE")
        print(f" Report: {self.final_report}")
        print("="*60)

    def _step_parse_excel(self) -> bool:
        print(f"\n[Step 1/2] Parsing Human Review: {self.excel_path.name}")
        
        try:
            parser = ExcelToAgentParser(str(self.excel_path))
            # Generate the JSONL intermediate file
            parser.generate_agent_directives(str(self.directives_jsonl))
            
            if not self.directives_jsonl.exists():
                print("    [!] Error: JSONL file was not created.")
                return False
            
            print(f"    [OK] Directives generated: {self.directives_jsonl}")
            return True
        except Exception as e:
            print(f"    [!] Exception during parsing: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _step_run_agent(self):
        print(f"\n[Step 2/2] Launching Fixer Agent")
        print(f"    Target Codebase: {self.codebase_root}")
        
        # Build LLM Config dictionary from CLI args
        llm_config = {
            "provider": self.args.llm_provider,
            "model": self.args.llm_model,
            "api_key": self.args.llm_api_key,
            "temperature": self.args.llm_temperature,
            "max_tokens": self.args.llm_max_tokens
        }

        if self.args.verbose:
            print(f"    [Configuration] LLM: {self.args.llm_provider} / {self.args.llm_model}")
            print(f"    [Configuration] Dry Run: {self.args.dry_run}")
            if self.args.llm_api_key:
                print("    [Configuration] API Key: Provided (masked)")
        
        try:
            # Initialize the Agent
            agent = CodebaseFixerAgent(
                codebase_root=str(self.codebase_root),
                directives_file=str(self.directives_jsonl),
                backup_dir=str(self.workspace_dir / "shelved_backups"),
                llm_config=llm_config,
                dry_run=self.args.dry_run,
                verbose=self.args.verbose
            )
            
            # Run the agent and generate the final audit report
            # FIX: Ensure final_report is passed as string to be safe
            agent.run_agent(report_filename=str(self.final_report))
            
        except Exception as e:
            print(f"    [!] Exception during agent execution: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()

# ==========================================
# Command Line Interface
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Automated Code Repair Workflow using Human Feedback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- File Paths ---
    # FIX: Removed nargs='?' because these are now named flags, not positionals.
    # Standard argparse behavior is cleaner here (user must provide value if flag is used).
    parser.add_argument(
        "--excel-file",
        default="out/detailed_code_review.xlsx", 
        help="Path to the reviewed Excel file"
    )
    parser.add_argument(
        "--codebase-path",
        default="codebase", 
        help="Root directory of the source code"
    )
    parser.add_argument(
        "--out-dir", 
        default="out", 
        help="Directory for backups/intermediate files"
    )

    # --- LLM Configuration ---
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "local", "mock", "qgenie"],
        default="qgenie",
        help="LLM Provider backend"
    )
    llm_group.add_argument("--llm-model", default="gemini-1.5-pro", help="Model identifier")
    llm_group.add_argument("--llm-api-key", help="API Key (overrides env vars)")
    llm_group.add_argument("--llm-max-tokens", type=int, default=15000, help="Token limit for context")
    llm_group.add_argument("--llm-temperature", type=float, default=0.1, help="Sampling temperature")

    # --- Safety & Debugging ---
    safe_group = parser.add_argument_group("Safety & Debugging")
    safe_group.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Simulate fixes without writing to disk"
    )
    safe_group.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging")
    safe_group.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Initialize and Run
    workflow = HumanInTheLoopWorkflow(args)
    workflow.execute()