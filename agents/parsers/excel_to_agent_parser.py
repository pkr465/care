import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional

class ExcelToAgentParser:
    """
    Parses a human-reviewed Excel report back into JSONL format 
    for the Automated Coding Agent.
    
    Logic:
    - Reads 'Feedback' and 'Constraints' columns.
    - Determines the 'action' (FIX, SKIP, RETRY) based on user input.
    - Preserves technical details (File, Line, Code) for the agent.
    """

    def __init__(self, excel_path: str):
        self.excel_path = excel_path

    def generate_agent_directives(self, output_jsonl: str = "agent_directives.jsonl"):
        """
        Converts Excel rows to JSONL.
        """
        if not os.path.exists(self.excel_path):
            print(f"[!] Error: File not found - {self.excel_path}")
            return

        print(f"[*] Reading human feedback from: {self.excel_path}")
        
        try:
            # Read the 'Analysis' sheet (skipping the Summary sheet)
            df = pd.read_excel(self.excel_path, sheet_name='Analysis', header=0)
            
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            
            # Check for required columns
            required = ["File", "Line", "Code", "Fixed_Code"]
            if not all(col in df.columns for col in required):
                print(f"[!] Error: Excel missing required columns. Found: {df.columns}")
                return

            directives = []
            stats = {"FIX": 0, "SKIP": 0, "CUSTOM": 0}

            for _, row in df.iterrows():
                # Extract Human Inputs
                feedback = str(row.get("Feedback", "")).strip()
                constraints = str(row.get("Constraints", "")).strip()
                
                # Handle NaN/Empty
                if feedback.lower() == "nan": feedback = ""
                if constraints.lower() == "nan": constraints = ""

                # Determine Action based on Feedback
                action = "FIX" # Default
                
                if feedback:
                    if any(x in feedback.upper() for x in ["SKIP", "IGNORE", "FALSE POSITIVE", "NO FIX"]):
                        action = "SKIP"
                    elif "RETRY" in feedback.upper() or constraints:
                        # If constraints exist, we treat it as a specialized fix
                        action = "FIX_WITH_CONSTRAINTS"
                    
                # Map to JSON structure
                entry = {
                    "file_path": row.get("File"),
                    "line_number": int(row.get("Line", 0)) if pd.notna(row.get("Line")) else 0,
                    "severity": row.get("Severity"),
                    "issue_type": row.get("Category"),
                    "bad_code_snippet": row.get("Code"),
                    "suggested_fix": row.get("Fixed_Code"), # The LLM's original suggestion
                    "human_feedback": feedback,
                    "human_constraints": constraints,
                    "action": action,
                    "run_id": str(row.get("S.No")) # Using S.No as a transient ID if RunID missing
                }

                directives.append(entry)
                
                # Update stats
                if action == "SKIP": stats["SKIP"] += 1
                elif action == "FIX_WITH_CONSTRAINTS": stats["CUSTOM"] += 1
                else: stats["FIX"] += 1

            # Write to JSONL
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for d in directives:
                    f.write(json.dumps(d) + '\n')

            print(f"[*] Successfully parsed {len(directives)} directives.")
            print(f"[*] Stats: {stats}")
            print(f"[*] Output saved to: {output_jsonl}")

        except Exception as e:
            print(f"[!] Parsing failed: {e}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Point this to your downloaded/edited Excel file
    INPUT_EXCEL = "out/detailed_code_review.xlsx"
    OUTPUT_JSONL = "out/human_guided_directives.jsonl"
    
    parser = ExcelToAgentParser(INPUT_EXCEL)
    parser.generate_agent_directives(OUTPUT_JSONL)