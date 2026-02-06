import subprocess
import os
import json
import logging
import shutil
from typing import Optional, Dict

# Configure Logger
logger = logging.getLogger(__name__)

class CCLSIngestion:
    """
    Handles the initialization and execution of CCLS indexing for C/C++ codebases.
    """

    def __init__(self, env_config: Optional[Dict[str, str]] = None):
        self.env = os.environ.copy()
        if env_config:
            self.env.update(env_config)

    def generate_ccls_config(self, project_root: str) -> str:
        """
        Generates the .ccls configuration file at the PROJECT ROOT.
        The .ccls file must sit alongside the source code for ccls to detect it.
        """
        # Generic safe defaults for C/C++
        # Removed specific --include=Global.h to prevent errors on generic repos
        ccls_file_content = r"""clang
%c -std=c11
%cpp -std=c++17
%ignore .*\.o$
%ignore .*\.d$
%ignore .*\.ko$
%ignore .*\.mod\.c
%ignore .*\.cmd
%ignore .*\.txt
%ignore .*\.log
%ignore .*\.bin
"""
        ccls_file_path = os.path.join(project_root, ".ccls")
        
        try:
            with open(ccls_file_path, "w") as f:
                f.write(ccls_file_content)
            logger.info(f"Generated .ccls config at {ccls_file_path}")
            return ccls_file_path
        except IOError as e:
            logger.warning(f"Failed to write .ccls file to project root (permission issue?): {e}")
            # We continue; ccls might still work with defaults or compile_commands.json
            return ""

    def run_indexing(self, project_root: str, output_dir: str, unique_project_prefix: str) -> bool:
        """
        Runs the ccls indexer.
        
        :param project_root: Absolute path to the source code.
        :param output_dir: Directory where the .ccls-cache will be stored.
        """
        # Ensure absolute paths
        abs_project_root = os.path.abspath(project_root)
        abs_output_dir = os.path.abspath(output_dir)

        # 1. Setup Cache Directory
        ccls_cache_dir = os.path.join(abs_output_dir, ".ccls-cache")
        if not os.path.exists(ccls_cache_dir):
            os.makedirs(ccls_cache_dir, exist_ok=True)

        # 2. Generate Config at Source Root
        self.generate_ccls_config(abs_project_root)

        # 3. Configure CCLS Cache Location
        # We tell ccls to store the cache in our output dir, not the source tree
        init_config = json.dumps({
            "cache": {
                "directory": ccls_cache_dir,
                "retainInMemory": 0 # Reduce memory usage for large repos
            }
        })

        # 4. Check Executable
        ccls_executable = self.env.get("CCLS_BIN_PATH", "ccls")
        if not shutil.which(ccls_executable):
            logger.error(f"CCLS executable '{ccls_executable}' not found in PATH.")
            return False

        ccls_cmd = [ccls_executable, "--index", abs_project_root, "--init", init_config]
        
        logger.info(f"Starting CCLS indexing for {abs_project_root}...")
        logger.debug(f"Command: {' '.join(ccls_cmd)}")
        
        try:
            # Run with capture_output=True equivalent to catch errors
            result = subprocess.run(
                ccls_cmd,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1200 # 20 minutes timeout
            )
            
            # Log specific success details if needed
            logger.info("CCLS Ingestion/Indexing Successful")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"CCLS indexing failed with exit code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            logger.error(f"STDOUT: {e.stdout}")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error(f"CCLS indexing timed out after 1200 seconds.")
            return False
            
        except Exception as e:
            logger.exception(f"Unexpected error during CCLS indexing: {e}")
            return False