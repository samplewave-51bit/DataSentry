"""
DataSentry Explainability - LLM Clients

Provides unified LLM generation interface supporting:
- IBM watsonx.ai Granite backend (live)
- Deterministic MockGraniteClient (100% offline, reproducible for CI/eval)
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class MockGraniteClient(LLMClient):
    """
    Deterministic offline mock mimicking IBM Granite (granite-13b-chat / granite-3-8b-instruct).
    Parses structured report input from the prompt and constructs faithful,
    grounded diagnostic explanations strictly citing finding IDs and report numbers.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Extract findings and metrics from prompt
        finding_matches = re.findall(r"(FND-[A-Z0-9-]+):\s*\[(critical|high|medium|low)\]\s*([^|\n]+)", prompt)
        dataset_match = re.search(r"Dataset:\s*([^\n]+)", prompt)
        status_match = re.search(r"Overall Gate Status:\s*([^\n]+)", prompt)

        dataset_name = dataset_match.group(1).strip() if dataset_match else "dataset_batch"
        overall_status = status_match.group(1).strip() if status_match else "WARN"

        lines = [
            f"# DataSentry Diagnostic Analysis: {dataset_name}",
            f"**Pipeline Gate Decision:** {overall_status}",
            "",
            "## 1. Executive Summary",
        ]

        if not finding_matches:
            lines.append(
                f"DataSentry scanned dataset '{dataset_name}' across quality, anomaly, and drift axes. "
                "No critical anomalies or distribution violations were detected. Batch is certified clean."
            )
            lines.extend([
                "",
                "## 2. Root Cause Analysis",
                "All evaluated features conform to reference specifications and statistical distributions.",
                "",
                "## 3. Recommended Remediation Plan",
                "- Proceed with model inference or feature store ingestion.",
            ])
            return "\n".join(lines)

        n_findings = len(finding_matches)
        lines.append(
            f"DataSentry monitoring engine evaluated dataset '{dataset_name}' and identified {n_findings} "
            f"diagnostic finding(s). The batch has been assigned status **{overall_status}**."
        )
        lines.extend(["", "## 2. Root Cause Analysis"])

        for f_id, sev, desc in finding_matches:
            clean_desc = desc.strip()
            lines.append(f"- **[{f_id}]** ({sev.upper()}): {clean_desc}")

        lines.extend(["", "## 3. Recommended Remediation Plan"])
        for f_id, sev, desc in finding_matches:
            if "duplicate" in desc.lower():
                lines.append(f"- Action for **[{f_id}]**: Execute deduplication pipeline to purge duplicate records.")
            elif "missing" in desc.lower() or "null" in desc.lower():
                lines.append(f"- Action for **[{f_id}]**: Quarantine batch and inspect upstream ingestion pipeline for null injection.")
            elif "schema" in desc.lower() or "column" in desc.lower():
                lines.append(f"- Action for **[{f_id}]**: Halt downstream inference and request schema synchronization from data engineering.")
            elif "drift" in desc.lower() or "coverage" in desc.lower():
                lines.append(f"- Action for **[{f_id}]**: Trigger model retraining or schedule human domain expert review for distribution shift.")
            else:
                lines.append(f"- Action for **[{f_id}]**: Isolate affected records and flag for auditor inspection.")

        return "\n".join(lines)


class WatsonxClient(LLMClient):
    """Live IBM watsonx.ai client utilizing Granite foundation models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WATSONX_API_KEY", "")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID", "")
        self.url = url or os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id = model_id or os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")

        if not self.api_key or not self.project_id:
            raise ValueError(
                "WatsonxClient requires WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Check if official SDK is installed
        try:
            from ibm_watsonx_ai.foundation_models import Model
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            parameters = {
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 800,
                GenParams.MIN_NEW_TOKENS: 10,
                GenParams.TEMPERATURE: 0.0,
            }
            credentials = {"url": self.url, "apikey": self.api_key}
            model = Model(
                model_id=self.model_id,
                params=parameters,
                credentials=credentials,
                project_id=self.project_id,
            )
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_text(prompt=full_prompt)
            return str(response)
        except ImportError:
            # Fallback to direct REST API
            import requests

            # Obtain IAM token from IBM Cloud
            iam_resp = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            iam_resp.raise_for_status()
            access_token = iam_resp.json()["access_token"]

            endpoint = f"{self.url}/ml/v1/text/generation?version=2023-05-29"
            payload = {
                "input": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "model_id": self.model_id,
                "project_id": self.project_id,
                "parameters": {"decoding_method": "greedy", "max_new_tokens": 800, "temperature": 0.0},
            }
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            api_resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            api_resp.raise_for_status()
            res_json = api_resp.json()
            return str(res_json["results"][0]["generated_text"])


def get_llm_client() -> LLMClient:
    """Factory selecting live watsonx.ai Granite or MockGraniteClient based on environment."""
    use_mock = os.getenv("DATASENTRY_USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not use_mock and api_key and project_id:
        try:
            return WatsonxClient()
        except Exception:
            pass

    return MockGraniteClient()
