import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Internal imports
# Assumes dependency_handler.py is in the python path
from dependency_builder.dependency_handler import DependencyFetcher

# Configure Logger
logger = logging.getLogger(__name__)

class DependencyService:
    """
    High-level service to orchestrate dependency fetching.
    Acts as the bridge between the Agent/Main and the lower-level DependencyFetcher.
    """

    def __init__(self, data_store_path: Optional[str] = None):
        self.data_store_path = data_store_path

    def _check_indexing_status(self, unique_project_prefix: str, output_dir: str) -> tuple[bool, Optional[Dict]]:
        """
        Verifies if the CCLS cache exists for the given project.
        """
        # Ensure we are looking at an absolute path to avoid CWD confusion
        abs_output_dir = os.path.abspath(output_dir)
        path_to_check = os.path.join(abs_output_dir, ".ccls-cache")

        if not Path(path_to_check).exists():
            msg = (
                f"Dependency Graph not generated yet for '{unique_project_prefix}'. "
                f"Ensure codebase ingestion/indexing is complete. Missing: {path_to_check}"
            )
            output_payload = {
                "message": msg,
                "data": {}
            }
            return False, output_payload
        
        return True, None

    def perform_fetch(
        self,
        project_root: str,
        output_dir: str,
        codebase_identifier: str,
        endpoint_type: str,
        file_name: str,
        function_name: Optional[str] = None,
        level: Optional[int] = 1,
        line: Optional[int] = None,
        character: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for Agents to request dependency information.
        """
        self.data_store_path = output_dir
        project_logger = logger
        
        # Clean the endpoint string (removed the buggy replace("", "") which did nothing)
        clean_endpoint = endpoint_type.strip()

        try:
            # 1. Check Indexing Status
            is_indexed, error_payload = self._check_indexing_status(codebase_identifier, output_dir)
            if not is_indexed:
                project_logger.warning(f"Indexing check failed: {error_payload['message']}")
                return error_payload

            # 2. Initialize Fetcher
            fetcher = DependencyFetcher()

            # 3. Process Request
            output_payload = fetcher.process_message(
                repo_path=project_root,
                data_store_path=output_dir,
                unique_project_prefix=codebase_identifier,
                endpoint_type=clean_endpoint,
                function_name=function_name,
                file_name=file_name,
                start=start,
                end=end,
                line=line,
                character=character,
                level=level,
                project_logger=project_logger
            )
            
            return output_payload

        except Exception as e:
            project_logger.error(f"Error in DependencyService: {e}", exc_info=True)
            return {
                "message": f"Critical error in dependency service: {str(e)}",
                "data": {}
            }

# --- Main Execution Block (for testing) ---
if __name__ == "__main__":
    service = DependencyService()
    TEST_ROOT = "/path/to/codebase"
    TEST_OUT = "./out"
    
    if os.path.exists(TEST_ROOT):
        # Example test
        pass