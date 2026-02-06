import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import logging

class EnvConfig:
    """
    Load, validate, normalize, and access environment variables
    for the DocxToMdConverter pipeline and related tools.
    """

    KEYS = [
        # DOCX→MD & Media Conversion
        "INPUTPATH",
        "OUTPUTPATH",
        "OUT_DIR",
        "FLAT_JSON_PATH",
        "REPORT_JSON_PATH",
        "WORD_DOC_FOLDER",
        "FORMAT",
        "TOC",
        "WITH_PANDOC",
        "WITH_WMF2SVG",
        "IMG_DIR",
        "HTML",
        "KEEP_IMG_DIMS",
        "RECALC_IMG_DIMS",
        "RECALC_MAX_DIMS",
        "VERBOSE",
        "DEBUG",
        # Legacy/output paths for backward compatibility
        "SOURCE_DIR",
        "GENERATED_MD_DIR",
        "DOC_FILES",
        "CODE_BASE_PATH",
        # QGenie/LangChain API keys and JSON/LLM configs
        "QGENIE_API_KEY",
        "EMBEDDING_MODEL",
        "LLM_MODEL", "STREAMLIT_MODEL","CONDING_LLM_MODEL",
        "CHAT_PROMPT_FILE_PATH",
        "LOG_LEVEL",
        "EMAIL_ID",
        #postgres db config
        "POSTGRES_CONNECTION",
        "POSTGRES_ADMIN_USERNAME",
        "POSTGRES_ADMIN_PASSWORD",
        "POSTGRES_COLLECTION",
        "POSTGRES_COLLECTION_TABLENAME",
        "POSTGRES_EMBEDDING_TABLENAME",
        "POSTGRES_STORE_NAME",
        "POSTGRES_USERNAME",
        "POSTGRES_DATABASE",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT"
    ]
    
    PATH_KEYS = [
        "INPUTPATH", "OUTPUTPATH", "OUT_DIR", "WORD_DOC_FOLDER",
        "IMG_DIR", "SOURCE_DIR", "GENERATED_MD_DIR", "PROMPT_FILE_PATH"
    ]

    BOOL_KEYS = [
        "TOC", "HTML", "KEEP_IMG_DIMS", "RECALC_IMG_DIMS", "VERBOSE", "DEBUG"
    ]
    
    INT_KEYS = [
        "RECALC_MAX_DIMS"
    ]

    @staticmethod
    def strip_quotes(value):
        """Strip single or double quotes from start/end."""
        if isinstance(value, str):
            return value.strip('\'"')
        return value

    @classmethod
    def get_default(cls, name):
        defaults = {
            "FORMAT": "docx",
            "EMBEDDING_MODEL": "qgenie",
            "LOG_LEVEL": "INFO",
            "RECALC_MAX_DIMS": 500,
            "TOC": False,
            "HTML": False,
            "KEEP_IMG_DIMS": False,
            "RECALC_IMG_DIMS": False,
            "VERBOSE": False,
            "DEBUG": False,
        }
        return defaults.get(name)

    def __init__(self):
        self.config = self.load_env()

    def load_env(self):
        """Loads, normalizes, and validates environment variables."""
        env_file = find_dotenv()
        print(f"Using .env: {env_file}")
        if env_file:
            load_dotenv(env_file)
        else:
            print("No .env found in the current or parent directories.")

        env_config = {}

        # Pull keys, strip, set default if needed
        for k in self.KEYS:
            raw = os.getenv(k, self.get_default(k))
            v = self.strip_quotes(raw)
            if v == "":
                v = None
            env_config[k] = v

        # Path normalization
        for k in self.PATH_KEYS:
            if env_config.get(k):
                env_config[k] = Path(env_config[k]).expanduser().resolve()

        # Boolean conversion
        for k in self.BOOL_KEYS:
            v = env_config.get(k)
            if isinstance(v, str):
                env_config[k] = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(v, bool):
                env_config[k] = v
            elif v is None:
                env_config[k] = self.get_default(k)
            else:
                env_config[k] = bool(v)

        # Integer conversion
        for k in self.INT_KEYS:
            v = env_config.get(k)
            try:
                if v is not None:
                    env_config[k] = int(v)
            except Exception:
                logging.warning(f"Could not parse int for key {k}: {v}")

        # Validation of required keys
        must_have = [
            "WORD_DOC_FOLDER", # document folder for batch
            "QGENIE_API_KEY",
        ]
        missing = [k for k in must_have if not env_config.get(k)]
        if missing:
            logging.error(f"Missing critical environment variables: {missing}")
            raise ValueError(f"Missing config: {missing}")

        # Enforce EMAIL_ID as a string (csv for lists)
        email_id = env_config.get("EMAIL_ID")
        if isinstance(email_id, list):
            env_config["EMAIL_ID"] = ",".join(str(e) for e in email_id)
        elif email_id is not None and not isinstance(email_id, str):
            env_config["EMAIL_ID"] = str(email_id)

        return env_config

    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

    def get_bool(self, key, default=False):
        """Get a boolean configuration value."""
        val = self.get(key, default)
        if isinstance(val, str):
            return val.lower() in ('1', 'true', 'yes', 'on')
        return bool(val)

    def get_int(self, key, default=0):
        """Get an int configuration value."""
        val = self.get(key, default)
        try:
            return int(val)
        except Exception:
            return default

# ===== USAGE =====
# env_config = EnvConfig()
# config_dict = env_config.config