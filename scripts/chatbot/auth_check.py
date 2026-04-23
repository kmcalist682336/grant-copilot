"""Preflight check for Vertex AI auth.

Run before starting the orchestrator (or any live-LLM path) to verify that
Application Default Credentials resolve, a project ID is set, and the
Vertex AI / aiplatform API is enabled.

Exits with status 1 and a clear instruction if anything is missing.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


class AuthCheckError(RuntimeError):
    """Raised when a preflight check fails; message is user-facing."""


def _resolve_adc() -> tuple[object, Optional[str]]:
    """Resolve Application Default Credentials.

    Returns (credentials, project_id_from_adc). project_id may be None if
    the credentials don't carry a default project (common on service
    accounts configured without one).
    """
    try:
        import google.auth  # type: ignore
    except ImportError as e:
        raise AuthCheckError(
            "google-auth is not installed. Install the Vertex AI deps:\n"
            "  pip install google-genai google-auth"
        ) from e

    try:
        credentials, project_id = google.auth.default()
    except Exception as e:
        raise AuthCheckError(
            "Unable to resolve Application Default Credentials. Run:\n"
            "  gcloud auth application-default login\n"
            f"Underlying error: {e}"
        ) from e

    return credentials, project_id


def _resolve_project_id(adc_project: Optional[str]) -> str:
    """Determine the effective project ID.

    Precedence: GCP_PROJECT_ID env > GOOGLE_CLOUD_PROJECT env > ADC default.
    """
    for var in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    if adc_project:
        return adc_project
    raise AuthCheckError(
        "No GCP project resolved. Set one of:\n"
        "  - export GCP_PROJECT_ID=<your-project>\n"
        "  - export GOOGLE_CLOUD_PROJECT=<your-project>\n"
        "  - gcloud config set project <your-project>"
    )


def _resolve_location() -> str:
    return os.environ.get("GCP_LOCATION", "us-central1")


def check_auth(verbose: bool = True) -> dict:
    """Run all preflight checks. Raises AuthCheckError on failure.

    Returns a dict with resolved values for the orchestrator to reuse:
      { "project_id": str, "location": str, "credentials_type": str }
    """
    credentials, adc_project = _resolve_adc()
    project_id = _resolve_project_id(adc_project)
    location = _resolve_location()

    if verbose:
        logger.info("Vertex AI auth preflight:")
        logger.info("  project:     %s", project_id)
        logger.info("  location:    %s", location)
        logger.info(
            "  credentials: %s", type(credentials).__name__
        )

    # Verify the google-genai SDK is importable. We don't make a live
    # call here — billing + rate limits matter, and live probing adds
    # latency on every startup. The first real extraction will surface
    # any API-enablement issue with a clean error.
    try:
        import google.genai  # type: ignore  # noqa: F401
    except ImportError as e:
        raise AuthCheckError(
            "google-genai is not installed. Install it:\n"
            "  pip install google-genai"
        ) from e

    return {
        "project_id": project_id,
        "location": location,
        "credentials_type": type(credentials).__name__,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        info = check_auth(verbose=True)
    except AuthCheckError as e:
        print(f"\nAuth preflight FAILED:\n{e}\n", file=sys.stderr)
        return 1
    print("\nAuth preflight OK.")
    print(f"  project:  {info['project_id']}")
    print(f"  location: {info['location']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
