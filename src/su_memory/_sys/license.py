"""
License Management Module
Handles license validation and capacity management
"""

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Default capacity limits
DEFAULT_CAPACITY = 1000
# v3.5.5 P0-5: License signing key (环境变量注入，不硬编码)
_LICENSE_HMAC_KEY = os.environ.get("SU_MEMORY_LICENSE_KEY", "").encode() if os.environ.get("SU_MEMORY_LICENSE_KEY") else None
# v3.6.1: 许可门禁全部移除 — 社区版即完整版
FREE_TIER_LIMITS = {
    "causal_hops": float('inf'),
    "prediction_per_day": None,  # 无限
    "vector_search": True,
    "explainability": True,
    "multi_session": True,
    "api_access": True,
}


@dataclass
class LicenseInfo:
    """License information"""
    license_type: str
    capacity: int | None
    license_key: str
    issued_to: str
    issued_at: str
    expires: str
    features: dict[str, bool]

    @property
    def is_expired(self) -> bool:
        """Check if license is expired"""
        if self.license_type == "on_premise":
            return False
        try:
            expiry = datetime.fromisoformat(self.expires.replace("Z", "+00:00"))
            return datetime.now() > expiry
        except Exception:
            return True

    @property
    def is_valid(self) -> bool:
        """Check if license is valid"""
        return not self.is_expired

    def get_capacity(self) -> int:
        """Get memory capacity"""
        return self.capacity if self.capacity else float('inf')

    def get_feature(self, feature: str, default: bool = False) -> bool:
        """Get feature availability"""
        return self.features.get(feature, default)

    def get_causal_hops(self) -> int:
        """Get max causal reasoning hops (v3.6.1: 所有版本无限制)"""
        return float('inf')

    def get_prediction_limit(self) -> int | None:
        """Get prediction calls per day (v3.6.1: 所有版本无限制)"""
        return None  # Unlimited


class LicenseManager:
    """
    License manager for su-memory SDK

    License file locations (in order of priority):
    1. ~/.su-memory/license.json (user home)
    2. ./.su-memory/license.json (current directory)
    3. Environment variable SU_MEMORY_LICENSE (base64 encoded)
    """

    LICENSE_DIR = Path.home() / ".su-memory"
    LICENSE_FILE = LICENSE_DIR / "license.json"
    LOCAL_LICENSE_FILE = Path(".su-memory/license.json")

    def __init__(self, license_path: str | None = None):
        """
        Initialize license manager

        Args:
            license_path: Custom license file path (optional)
        """
        self._license_info: LicenseInfo | None = None
        self._custom_path = Path(license_path) if license_path else None

    def load_license(self) -> LicenseInfo | None:
        """
        Load license from file or environment

        Returns:
            LicenseInfo if valid license found, None otherwise
        """
        # Check custom path first
        if self._custom_path and self._custom_path.exists():
            return self._parse_license_file(self._custom_path)

        # Check local directory
        if self.LOCAL_LICENSE_FILE.exists():
            return self._parse_license_file(self.LOCAL_LICENSE_FILE)

        # Check user home directory
        if self.LICENSE_FILE.exists():
            return self._parse_license_file(self.LICENSE_FILE)

        # Check environment variable
        import os
        encoded = os.getenv("SU_MEMORY_LICENSE")
        if encoded:
            try:
                import base64
                decoded = base64.b64decode(encoded).decode()
                data = json.loads(decoded)
                if not self._verify_license_signature(data):
                    print("Warning: Environment license signature verification failed")
                    return None
                return self._create_license_info(data)
            except Exception:
                pass

        return None

    def _parse_license_file(self, path: Path) -> LicenseInfo | None:
        """Parse license file with signature verification (v3.5.5 P0-5)"""
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not self._verify_license_signature(data):
                print("Warning: License signature verification failed")
                return None
            return self._create_license_info(data)
        except Exception as e:
            print(f"Warning: Failed to parse license file: {e}")
            return None

    def _verify_license_signature(self, data: dict) -> bool:
        """
        v3.5.5 P0-5修复: 验证许可证 HMAC 签名。

        许可证 JSON 中必须包含 'signature' 字段，其值为
        HMAC-SHA256(license_key:capacity:expires:issued_to, SU_MEMORY_LICENSE_KEY)。
        如果未设置 SU_MEMORY_LICENSE_KEY 环境变量，跳过验证 (向后兼容开发环境)。
        """
        if _LICENSE_HMAC_KEY is None:
            return True  # 开发环境：无签名密钥时允许通过

        expected_sig = data.get("signature", "")
        if not expected_sig:
            return False

        payload = f"{data.get('license_key','')}:{data.get('capacity','')}:{data.get('expires','')}:{data.get('issued_to','')}"
        computed = hmac.new(_LICENSE_HMAC_KEY, payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(computed, expected_sig)

    def _create_license_info(self, data: dict) -> LicenseInfo:
        """Create LicenseInfo from data"""
        return LicenseInfo(
            license_type=data.get("license_type", "community"),
            capacity=data.get("capacity", DEFAULT_CAPACITY),
            license_key=data.get("license_key", ""),
            issued_to=data.get("issued_to", ""),
            issued_at=data.get("issued_at", ""),
            expires=data.get("expires", ""),
            features=data.get("features", FREE_TIER_LIMITS),
        )

    @property
    def license_info(self) -> LicenseInfo:
        """Get current license info (lazy load)"""
        if self._license_info is None:
            self._license_info = self.load_license()
        return self._license_info

    @property
    def is_licensed(self) -> bool:
        """Check if valid license exists"""
        info = self.license_info
        return info is not None and info.is_valid

    @property
    def license_type(self) -> str:
        """Get license type"""
        info = self.license_info
        return info.license_type if info else "community"

    def get_capacity(self) -> int:
        """Get memory capacity"""
        info = self.license_info
        return info.get_capacity() if info else DEFAULT_CAPACITY

    def check_capacity(self, current_count: int) -> bool:
        """
        Check if adding more memories would exceed capacity

        Args:
            current_count: Current number of memories

        Returns:
            True if within capacity, False otherwise
        """
        return current_count < self.get_capacity()

    def get_causal_hops(self) -> int:
        """Get max causal reasoning hops"""
        info = self.license_info
        return info.get_causal_hops() if info else 3

    def get_prediction_limit(self) -> int | None:
        """Get prediction calls per day"""
        info = self.license_info
        return info.get_prediction_limit() if info else 3

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled

        Args:
            feature: Feature name (vector_search, causal_reasoning, etc.)

        Returns:
            True if feature is enabled
        """
        info = self.license_info
        if not info:
            return FREE_TIER_LIMITS.get(feature, False)
        return info.get_feature(feature, FREE_TIER_LIMITS.get(feature, False))

    def get_status(self) -> dict[str, Any]:
        """
        Get license status summary

        Returns:
            Dictionary with license status
        """
        info = self.license_info
        if not info:
            return {
                "licensed": False,
                "type": "community",
                "capacity": DEFAULT_CAPACITY,
                "features": FREE_TIER_LIMITS,
                "message": "Using free Community version"
            }

        return {
            "licensed": True,
            "type": info.license_type,
            "capacity": info.get_capacity(),
            "expires": info.expires,
            "license_key": info.license_key[:8] + "..." if info.license_key else None,
            "features": info.features,
            "message": f"Licensed: {info.license_type}"
        }

    @staticmethod
    def create_license_file(
        license_key: str,
        license_type: str,
        capacity: int | None,
        issued_to: str,
        expires: str,
        features: dict[str, bool],
        output_path: str | None = None
    ) -> Path:
        """
        Create a license file (for offline activation)

        Args:
            license_key: License key
            license_type: License type (community/pro/enterprise/on_premise)
            capacity: Memory capacity (None for unlimited)
            issued_to: Customer email
            expires: Expiry date (ISO format)
            features: Feature flags
            output_path: Output file path

        Returns:
            Path to created license file
        """
        license_data = {
            "version": "1.0",
            "license_key": license_key,
            "license_type": license_type,
            "capacity": capacity,
            "issued_to": issued_to,
            "issued_at": datetime.now().isoformat(),
            "expires": expires,
            "features": features,
        }

        # Determine output path
        if output_path:
            path = Path(output_path)
        else:
            # Create in user home directory
            LicenseManager.LICENSE_DIR.mkdir(parents=True, exist_ok=True)
            path = LicenseManager.LICENSE_FILE

        # Write license file
        path.write_text(json.dumps(license_data, indent=2, ensure_ascii=False), encoding='utf-8')

        return path

    @staticmethod
    def install_license(license_content: str) -> bool:
        """
        Install license from content string

        Args:
            license_content: License JSON content or base64 encoded string

        Returns:
            True if successful
        """
        try:
            # Try parsing as JSON first
            try:
                data = json.loads(license_content)
            except Exception:
                # Try as base64
                import base64
                data = json.loads(base64.b64decode(license_content).decode())

            # Validate required fields
            required = ["license_key", "license_type"]
            if not all(k in data for k in required):
                print("Error: Invalid license format")
                return False

            # v3.5.5 P0-5: 验证许可证签名
            mgr = LicenseManager()
            if not mgr._verify_license_signature(data):
                print("Error: License signature verification failed")
                return False

            # Save to user home
            LicenseManager.LICENSE_DIR.mkdir(parents=True, exist_ok=True)
            LicenseManager.LICENSE_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

            print(f"License installed successfully: {data['license_key'][:16]}...")
            return True

        except Exception as e:
            print(f"Failed to install license: {e}")
            return False


# Global license manager instance
# v3.5.5 P1-7: DCL singleton for license manager
_license_manager: LicenseManager | None = None
_license_manager_lock = __import__('threading').Lock()


def get_license_manager() -> LicenseManager:
    """Get global license manager instance (thread-safe DCL, v3.5.5 P1-7)"""
    global _license_manager
    if _license_manager is None:
        with _license_manager_lock:
            if _license_manager is None:
                _license_manager = LicenseManager()
    return _license_manager


def check_license() -> dict[str, Any]:
    """Quick license status check"""
    return get_license_manager().get_status()
