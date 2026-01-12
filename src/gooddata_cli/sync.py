"""Sync GoodData artifacts to local cache for offline access.

This module syncs LDM, analytics model, and catalog data from GoodData
to local JSON files in each customer's project directory.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gooddata_cli.sdk import get_sdk

CONFIG_PATH = Path.home() / ".config" / "gooddata" / "workspaces.yaml"


def load_config() -> dict[str, Any]:
    """Load workspace configuration from ~/.config/gooddata/workspaces.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}\n"
            "Create it with customer workspace mappings."
        )

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_customers() -> dict[str, dict[str, Any]]:
    """Get all configured customers."""
    config = load_config()
    return config.get("customers", {})


def get_sync_settings() -> dict[str, Any]:
    """Get sync settings from config."""
    config = load_config()
    return config.get("sync", {})


def sync_workspace(
    workspace_id: str,
    output_dir: Path,
    workspace_name: str | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Sync GoodData artifacts for a single workspace.

    Args:
        workspace_id: GoodData workspace ID
        output_dir: Directory to save JSON files
        workspace_name: Optional workspace name for logging
        artifacts: List of artifacts to sync (ldm, analytics_model, catalog)

    Returns:
        Dict with sync results
    """
    sdk = get_sdk()
    output_dir.mkdir(parents=True, exist_ok=True)

    if artifacts is None:
        artifacts = ["ldm", "analytics_model", "catalog"]

    results = {
        "workspace_id": workspace_id,
        "workspace_name": workspace_name,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }

    display_name = workspace_name or workspace_id

    # Sync LDM
    if "ldm" in artifacts:
        try:
            ldm = sdk.catalog_workspace_content.get_declarative_ldm(workspace_id)
            ldm_dict = ldm.to_dict()

            # Add summary for quick reference
            datasets = ldm_dict.get("ldm", {}).get("datasets", [])
            date_instances = ldm_dict.get("ldm", {}).get("dateInstances", [])

            ldm_output = {
                "workspace_id": workspace_id,
                "synced_at": results["synced_at"],
                "summary": {
                    "dataset_count": len(datasets),
                    "date_instance_count": len(date_instances),
                },
                "ldm": ldm_dict.get("ldm", {}),
            }

            with open(output_dir / "ldm.json", "w") as f:
                json.dump(ldm_output, f, indent=2, default=str)

            results["artifacts"]["ldm"] = {
                "success": True,
                "datasets": len(datasets),
                "date_instances": len(date_instances),
            }
            print(f"  [LDM] {display_name}: {len(datasets)} datasets, {len(date_instances)} date instances")
        except Exception as e:
            results["artifacts"]["ldm"] = {"success": False, "error": str(e)}
            print(f"  [LDM] {display_name}: ERROR - {e}")

    # Sync Analytics Model
    if "analytics_model" in artifacts:
        try:
            am = sdk.catalog_workspace_content.get_declarative_analytics_model(workspace_id)
            am_dict = am.to_dict()

            analytics = am_dict.get("analytics", {})

            am_output = {
                "workspace_id": workspace_id,
                "synced_at": results["synced_at"],
                "summary": {
                    "metric_count": len(analytics.get("metrics", [])),
                    "insight_count": len(analytics.get("visualizationObjects", [])),
                    "dashboard_count": len(analytics.get("analyticalDashboards", [])),
                    "filter_context_count": len(analytics.get("filterContexts", [])),
                },
                "analytics": analytics,
            }

            with open(output_dir / "analytics_model.json", "w") as f:
                json.dump(am_output, f, indent=2, default=str)

            results["artifacts"]["analytics_model"] = {
                "success": True,
                "metrics": am_output["summary"]["metric_count"],
                "insights": am_output["summary"]["insight_count"],
                "dashboards": am_output["summary"]["dashboard_count"],
            }
            print(
                f"  [Analytics] {display_name}: "
                f"{am_output['summary']['metric_count']} metrics, "
                f"{am_output['summary']['insight_count']} insights, "
                f"{am_output['summary']['dashboard_count']} dashboards"
            )
        except Exception as e:
            results["artifacts"]["analytics_model"] = {"success": False, "error": str(e)}
            print(f"  [Analytics] {display_name}: ERROR - {e}")

    # Sync Catalog
    if "catalog" in artifacts:
        try:
            catalog = sdk.catalog_workspace_content.get_full_catalog(workspace_id)

            catalog_output = {
                "workspace_id": workspace_id,
                "synced_at": results["synced_at"],
                "summary": {
                    "dataset_count": len(catalog.datasets),
                    "metric_count": len(catalog.metrics),
                },
                "datasets": [
                    {
                        "id": ds.id,
                        "title": ds.title,
                        "attributes": [
                            {"id": a.id, "title": a.title}
                            for a in ds.attributes
                        ],
                        "facts": [
                            {"id": f.id, "title": f.title}
                            for f in ds.facts
                        ],
                    }
                    for ds in catalog.datasets
                ],
                "metrics": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "format": getattr(m, "format", None),
                    }
                    for m in catalog.metrics
                ],
            }

            with open(output_dir / "catalog.json", "w") as f:
                json.dump(catalog_output, f, indent=2, default=str)

            results["artifacts"]["catalog"] = {
                "success": True,
                "datasets": len(catalog.datasets),
                "metrics": len(catalog.metrics),
            }
            print(
                f"  [Catalog] {display_name}: "
                f"{len(catalog.datasets)} datasets, "
                f"{len(catalog.metrics)} metrics"
            )
        except Exception as e:
            results["artifacts"]["catalog"] = {"success": False, "error": str(e)}
            print(f"  [Catalog] {display_name}: ERROR - {e}")

    # Save sync metadata
    with open(output_dir / "_sync_metadata.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def sync_customer(
    customer_name: str,
    include_children: bool = True,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Sync all workspaces for a customer.

    Args:
        customer_name: Customer key from config (tpp, dlg, danceone)
        include_children: Whether to sync child workspaces
        artifacts: List of artifacts to sync

    Returns:
        Dict with sync results for all workspaces
    """
    customers = get_customers()

    if customer_name not in customers:
        raise ValueError(
            f"Customer '{customer_name}' not found in config. "
            f"Available: {list(customers.keys())}"
        )

    customer = customers[customer_name]
    project_path = Path(customer["project_path"])
    output_base = project_path / customer.get("output_dir", "docs-local/gooddata")

    results = {
        "customer": customer_name,
        "project_path": str(project_path),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "workspaces": {},
    }

    print(f"\nSyncing {customer_name.upper()}...")
    print(f"  Project: {project_path}")

    # Sync parent workspace
    workspace_id = customer["workspace_id"]
    workspace_name = customer.get("workspace_name", workspace_id)

    results["workspaces"]["parent"] = sync_workspace(
        workspace_id=workspace_id,
        output_dir=output_base,
        workspace_name=workspace_name,
        artifacts=artifacts,
    )

    # Sync child workspace if configured
    if include_children and customer.get("child_workspace_id"):
        child_id = customer["child_workspace_id"]
        child_name = customer.get("child_workspace_name", child_id)

        results["workspaces"]["child"] = sync_workspace(
            workspace_id=child_id,
            output_dir=output_base / "child",
            workspace_name=child_name,
            artifacts=artifacts,
        )

    return results


def sync_all(
    include_children: bool = True,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Sync all configured customers.

    Args:
        include_children: Whether to sync child workspaces
        artifacts: List of artifacts to sync

    Returns:
        Dict with sync results for all customers
    """
    customers = get_customers()
    settings = get_sync_settings()

    if artifacts is None:
        artifacts = settings.get("artifacts", ["ldm", "analytics_model", "catalog"])

    if include_children is None:
        include_children = settings.get("include_children", True)

    results = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "customers": {},
    }

    print("=" * 60)
    print("GoodData Sync - All Customers")
    print("=" * 60)

    for customer_name in customers:
        try:
            results["customers"][customer_name] = sync_customer(
                customer_name=customer_name,
                include_children=include_children,
                artifacts=artifacts,
            )
        except Exception as e:
            results["customers"][customer_name] = {
                "error": str(e),
                "success": False,
            }
            print(f"\n{customer_name.upper()}: ERROR - {e}")

    print("\n" + "=" * 60)
    print("Sync complete!")
    print("=" * 60)

    return results


def get_sync_status() -> dict[str, Any]:
    """Get sync status for all configured customers.

    Returns:
        Dict with last sync times and artifact counts for each customer
    """
    customers = get_customers()
    status = {}

    for customer_name, customer in customers.items():
        project_path = Path(customer["project_path"])
        output_base = project_path / customer.get("output_dir", "docs-local/gooddata")
        metadata_path = output_base / "_sync_metadata.json"

        customer_status = {
            "project_path": str(project_path),
            "output_dir": str(output_base),
            "parent": None,
            "child": None,
        }

        # Check parent workspace
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            customer_status["parent"] = {
                "synced_at": metadata.get("synced_at"),
                "artifacts": metadata.get("artifacts", {}),
            }
        else:
            customer_status["parent"] = {"synced_at": None, "status": "not synced"}

        # Check child workspace
        child_metadata_path = output_base / "child" / "_sync_metadata.json"
        if child_metadata_path.exists():
            with open(child_metadata_path) as f:
                metadata = json.load(f)
            customer_status["child"] = {
                "synced_at": metadata.get("synced_at"),
                "artifacts": metadata.get("artifacts", {}),
            }
        elif customer.get("child_workspace_id"):
            customer_status["child"] = {"synced_at": None, "status": "not synced"}

        status[customer_name] = customer_status

    return status
