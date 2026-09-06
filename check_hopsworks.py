"""
Quick connectivity check for Hopsworks credentials and project access.

Run:
    python check_hopsworks.py
"""

from src.config import HOPSWORKS_HOST, HOPSWORKS_PROJECT
from src.hopsworks_client import login_to_hopsworks


def main() -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    print("✅ Hopsworks login successful")
    print(f"Host: {HOPSWORKS_HOST}")
    print(f"Project: {HOPSWORKS_PROJECT}")
    print(f"Feature Store: {type(fs).__name__}")


if __name__ == "__main__":
    main()
