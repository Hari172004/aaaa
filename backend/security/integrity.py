"""
integrity.py — Bot Code Integrity & Hash Verification
=====================================================
Calculates SHA-256 hashes of all core Python files to detect unauthorized 
modifications at runtime.
"""

import os
import hashlib
import logging
import json
from typing import Dict, List, Optional

logger = logging.getLogger("agniv.security.integrity")

# Files that should NEVER change without regenerating the hash map
CORE_FILES = [
    "core.py",
    "risk_manager.py",
    "gold_risk_manager.py",
    "logger.py",
    "run_bot.py",
    "backend/security/encryption.py",
    "backend/security/auth.py",
]

# Directories where all .py files must be monitored
PROTECTED_DIRS = [
    "strategies",
    "filters",
    "broker",
    "alerts",
]

HASH_MAP_FILE = "backend/security/checksums.json"

class IntegrityManager:
    @staticmethod
    def calculate_file_hash(file_path: str) -> Optional[str]:
        """Returns the SHA-256 hash of a file."""
        if not os.path.exists(file_path):
            return None
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"[Integrity] Error hashing {file_path}: {e}")
            return None

    def generate_current_hashes(self, root_dir: str = ".") -> Dict[str, str]:
        """Scans core files and protected directories to build a new hash map."""
        hashes = {}
        
        # 1. Individual core files
        for f in CORE_FILES:
            path = os.path.join(root_dir, f)
            h = self.calculate_file_hash(path)
            if h:
                hashes[f] = h
        
        # 2. Protected directories
        for d in PROTECTED_DIRS:
            d_path = os.path.join(root_dir, d)
            if not os.path.isdir(d_path):
                continue
            for root, _, files in os.walk(d_path):
                for f in files:
                    if f.endswith(".py"):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
                        h = self.calculate_file_hash(full_path)
                        if h:
                            hashes[rel_path] = h
                            
        return hashes

    def save_checksums(self, hashes: Dict[str, str]):
        """Persists the checksums to a JSON file (Admin only)."""
        try:
            with open(HASH_MAP_FILE, "w") as f:
                json.dump(hashes, f, indent=4)
            logger.info(f"[Integrity] Checksums saved to {HASH_MAP_FILE}")
        except Exception as e:
            logger.error(f"[Integrity] Failed to save checksums: {e}")

    def load_checksums(self) -> Dict[str, str]:
        """Loads the saved checksums from the JSON file."""
        if not os.path.exists(HASH_MAP_FILE):
            logger.warning("[Integrity] No checksums.json found. System unprotected.")
            return {}
        try:
            with open(HASH_MAP_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[Integrity] Failed to load checksums: {e}")
            return {}

    def verify_integrity(self, root_dir: str = ".") -> List[str]:
        """
        Compares current file hashes against the saved map.
        Returns a list of files that have been modified or are missing.
        """
        saved = self.load_checksums()
        if not saved:
            return [] # Cannot verify without a map

        violations = []
        current = self.generate_current_hashes(root_dir)
        
        # Check for changed or missing files
        for path, saved_h in saved.items():
            curr_h = current.get(path)
            if curr_h != saved_h:
                logger.error(f"[Integrity] 🚨 TAMPER DETECTED: {path}")
                violations.append(path)
        
        return violations

# Global singleton
checker = IntegrityManager()
