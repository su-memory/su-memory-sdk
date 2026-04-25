#!/usr/bin/env python3
"""
su-memory SDK License Installation Script

Usage:
    python install_license.py                    # Interactive mode
    python install_license.py --license-key XXX  # Non-interactive mode
    python install_license.py --file license.json  # From file
    python install_license.py --uninstall       # Remove license
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from su_memory._sys.license import LicenseManager, check_license


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("  su-memory SDK License Installer")
    print("=" * 60)
    print()


def print_current_status():
    """Print current license status"""
    status = check_license()
    print("Current Status:")
    print(f"  License Type: {status['type']}")
    print(f"  Capacity: {status['capacity']:,}" + (" (unlimited)" if status['capacity'] == float('inf') else ""))
    print(f"  Licensed: {'Yes' if status['licensed'] else 'No'}")
    if status.get('expires'):
        print(f"  Expires: {status['expires']}")
    print()


def interactive_install():
    """Interactive license installation"""
    print_banner()
    print_current_status()
    
    print("Enter your license key (or press Ctrl+C to cancel):")
    license_key = input("> ").strip()
    
    if not license_key:
        print("Error: License key cannot be empty")
        return False
    
    # For demo purposes, create a basic license
    # In production, this would verify against a server
    license_data = {
        "version": "1.0",
        "license_key": license_key,
        "license_type": detect_license_type(license_key),
        "capacity": detect_capacity(license_key),
        "issued_to": "user@example.com",
        "issued_at": "2026-04-25T00:00:00",
        "expires": "2027-04-25",
        "features": {
            "vector_search": True,
            "causal_reasoning": True,
            "temporal_prediction": True,
            "explainability": True,
            "multi_session": True,
            "api_access": False,
        }
    }
    
    # Save license
    LicenseManager.LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    LicenseManager.LICENSE_FILE.write_text(
        json.dumps(license_data, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    print("\n✅ License installed successfully!")
    print(f"   Key: {license_key[:16]}...")
    print(f"   Type: {license_data['license_type']}")
    print(f"   Location: {LicenseManager.LICENSE_FILE}")
    
    return True


def install_from_file(file_path: str):
    """Install license from file"""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        
        # Validate required fields
        required = ["license_key", "license_type"]
        if not all(k in data for k in required):
            print("Error: Invalid license file format")
            return False
        
        # Save license
        LicenseManager.LICENSE_DIR.mkdir(parents=True, exist_ok=True)
        LicenseManager.LICENSE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        print(f"✅ License installed from: {file_path}")
        print(f"   Key: {data['license_key'][:16]}...")
        print(f"   Type: {data['license_type']}")
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to install license: {e}")
        return False


def install_from_key(license_key: str):
    """Install license from key"""
    print_banner()
    
    if not license_key:
        print("Error: License key cannot be empty")
        return False
    
    # Create license data
    license_data = {
        "version": "1.0",
        "license_key": license_key,
        "license_type": detect_license_type(license_key),
        "capacity": detect_capacity(license_key),
        "issued_to": "licensed_user",
        "issued_at": "2026-04-25T00:00:00",
        "expires": "2027-04-25",
        "features": {
            "vector_search": True,
            "causal_reasoning": True,
            "temporal_prediction": True,
            "explainability": True,
            "multi_session": True,
            "api_access": False,
        }
    }
    
    # Save license
    LicenseManager.LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    LicenseManager.LICENSE_FILE.write_text(
        json.dumps(license_data, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    print("✅ License installed successfully!")
    print(f"   Key: {license_key[:16]}...")
    print(f"   Type: {license_data['license_type']}")
    print(f"   Location: {LicenseManager.LICENSE_FILE}")
    
    return True


def uninstall_license():
    """Remove installed license"""
    print_banner()
    
    if not LicenseManager.LICENSE_FILE.exists():
        print("No license file found to remove.")
        return True
    
    try:
        LicenseManager.LICENSE_FILE.unlink()
        print("✅ License uninstalled successfully!")
        return True
    except Exception as e:
        print(f"Error: Failed to uninstall license: {e}")
        return False


def detect_license_type(key: str) -> str:
    """Detect license type from key prefix"""
    key_upper = key.upper()
    if "ONPREMISE" in key_upper or "ON_PREMISE" in key_upper:
        return "on_premise"
    elif "ENTERPRISE" in key_upper or "ENT" in key_upper:
        return "enterprise"
    elif "PRO" in key_upper:
        return "pro"
    else:
        return "community"


def detect_capacity(key: str) -> int:
    """Detect capacity from license type"""
    license_type = detect_license_type(key)
    capacities = {
        "community": 1000,
        "pro": 10000,
        "enterprise": 100000,
        "on_premise": None,  # Unlimited
    }
    return capacities.get(license_type, 1000)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="su-memory SDK License Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_license.py                        # Interactive mode
  python install_license.py --license-key SM-PRO-XXXX-XXXX
  python install_license.py --file license.json
  python install_license.py --status              # Check current status
  python install_license.py --uninstall           # Remove license
        """
    )
    
    parser.add_argument("--license-key", "-k", help="License key")
    parser.add_argument("--file", "-f", help="License file path")
    parser.add_argument("--status", "-s", action="store_true", help="Show license status")
    parser.add_argument("--uninstall", "-u", action="store_true", help="Uninstall license")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle commands
    if args.status:
        print_current_status()
        return 0
    
    if args.uninstall:
        uninstall_license()
        return 0
    
    if args.file:
        success = install_from_file(args.file)
        return 0 if success else 1
    
    if args.license_key:
        success = install_from_key(args.license_key)
        return 0 if success else 1
    
    # Interactive mode
    success = interactive_install()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
