#!/usr/bin/env python3
"""MCP Server for GoodData SDK CLI.

This exposes GoodData operations as MCP tools for use with Claude Code.
All operations are READ-ONLY.
"""

import json
import os
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("gooddata")


def _load_env():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _get_sdk():
    """Get GoodData SDK instance."""
    _load_env()
    from gooddata_sdk import GoodDataSdk

    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    return GoodDataSdk.create(host, token)


def _get_workspace_id(workspace_id: str | None = None) -> str:
    """Get workspace ID from argument or environment."""
    if workspace_id:
        return workspace_id

    _load_env()
    env_workspace = os.getenv("GOODDATA_WORKSPACE")

    if not env_workspace:
        raise ValueError("No workspace_id provided and GOODDATA_WORKSPACE not set")

    return env_workspace


# =============================================================================
# LIST TOOLS (Read-Only)
# =============================================================================


@mcp.tool()
def list_workspaces() -> str:
    """List all available GoodData workspaces.

    Returns a JSON array of workspaces with their IDs and names.
    """
    sdk = _get_sdk()
    workspaces = sdk.catalog_workspace.list_workspaces()

    result = [{"id": ws.id, "name": ws.name} for ws in workspaces]
    return json.dumps(result, indent=2)


@mcp.tool()
def list_insights(workspace_id: str | None = None) -> str:
    """List all insights (visualizations) in a workspace.

    Args:
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns a JSON array of insights with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    result = [{"id": viz.id, "title": viz.title} for viz in am.analytics.visualization_objects]
    return json.dumps(result, indent=2)


@mcp.tool()
def list_dashboards(workspace_id: str | None = None) -> str:
    """List all dashboards in a workspace.

    Args:
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns a JSON array of dashboards with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    result = [{"id": db.id, "title": db.title} for db in am.analytics.analytical_dashboards]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_dashboard_insights(dashboard_id: str, workspace_id: str | None = None) -> str:
    """Get all insights (visualizations) contained in a specific dashboard.

    Args:
        dashboard_id: The dashboard ID to get insights from.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns a JSON object with dashboard info and an array of insights with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    # Find the dashboard
    dashboard = None
    for db in am.analytics.analytical_dashboards:
        if db.id == dashboard_id:
            dashboard = db
            break

    if not dashboard:
        return json.dumps({"error": f"Dashboard '{dashboard_id}' not found"})

    # Build a lookup of all visualization objects by ID
    viz_lookup = {viz.id: viz.title for viz in am.analytics.visualization_objects}

    # Extract insight IDs from dashboard layout
    insight_ids = []
    content = dashboard.content
    layout = content.get("layout", {})
    sections = layout.get("sections", [])

    for section in sections:
        items = section.get("items", [])
        for item in items:
            widget = item.get("widget", {})
            if widget.get("type") == "insight":
                insight_ref = widget.get("insight", {})
                identifier = insight_ref.get("identifier", {})
                if identifier.get("type") == "visualizationObject":
                    insight_id = identifier.get("id")
                    if insight_id:
                        insight_ids.append(
                            {
                                "id": insight_id,
                                "title": viz_lookup.get(insight_id, widget.get("title", "")),
                                "widget_title": widget.get("title", ""),
                            }
                        )

    result = {
        "dashboard_id": dashboard_id,
        "dashboard_title": dashboard.title,
        "insights": insight_ids,
        "insight_count": len(insight_ids),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def list_metrics(workspace_id: str | None = None) -> str:
    """List all metrics in a workspace.

    Args:
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns a JSON array of metrics with all available properties.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)

    result = [
        {
            "id": m.id,
            "title": m.title,
            "format": getattr(m, "format", None),
            "is_hidden": getattr(m, "is_hidden", None),
            "obj_id": getattr(m, "obj_id", None),
            "json_api_attributes": getattr(m, "json_api_attributes", None),
            "json_api_related_entities_data": getattr(m, "json_api_related_entities_data", None),
            "json_api_related_entities_side_loads": getattr(
                m, "json_api_related_entities_side_loads", None
            ),
            "json_api_relationships": getattr(m, "json_api_relationships", None),
            "json_api_side_loads": getattr(m, "json_api_side_loads", None),
        }
        for m in catalog.metrics
    ]
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def list_datasets(workspace_id: str | None = None) -> str:
    """List all datasets in a workspace.

    Args:
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns a JSON array of datasets with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)

    result = [{"id": ds.id, "title": ds.title} for ds in catalog.datasets]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_logical_data_model(
    workspace_id: str | None = None,
    output_path: str | None = None,
) -> str:
    """Get the logical data model (LDM) for a workspace.

    The LDM contains all datasets, attributes, labels, facts, and their relationships.
    This is useful for understanding the data structure and for documentation.

    Args:
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.
        output_path: Optional file path to save the LDM. If provided, saves as JSON file.

    Returns:
        JSON containing the full logical data model structure, or path to saved file.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    # Get the declarative LDM
    ldm = sdk.catalog_workspace_content.get_declarative_ldm(ws_id)
    ldm_dict = ldm.to_dict()

    # Build a summary
    datasets = ldm_dict.get("ldm", {}).get("datasets", [])
    date_instances = ldm_dict.get("ldm", {}).get("dateInstances", [])

    summary = {
        "workspace_id": ws_id,
        "dataset_count": len(datasets),
        "date_instance_count": len(date_instances),
        "datasets": [],
    }

    for ds in datasets:
        ds_summary = {
            "id": ds.get("id"),
            "title": ds.get("title"),
            "attribute_count": len(ds.get("attributes", [])),
            "fact_count": len(ds.get("facts", [])),
            "reference_count": len(ds.get("references", [])),
        }
        summary["datasets"].append(ds_summary)

    if output_path:
        # Save full LDM to file
        output_dir = Path(output_path).parent
        if output_dir and str(output_dir) != ".":
            output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(ldm_dict, f, indent=2, default=str)

        return json.dumps(
            {
                "success": True,
                "path": os.path.abspath(output_path),
                "summary": summary,
            },
            indent=2,
        )

    # Return summary with full LDM
    return json.dumps(
        {
            "summary": summary,
            "ldm": ldm_dict,
        },
        indent=2,
        default=str,
    )


# =============================================================================
# USER & GROUP TOOLS (Read-Only)
# =============================================================================


@mcp.tool()
def list_users() -> str:
    """List all users in the GoodData organization.

    Returns a JSON array of users with their IDs and names.
    """
    sdk = _get_sdk()
    users = sdk.catalog_user.list_users()

    result = [
        {
            "id": u.id,
            "name": getattr(u, "name", None),
            "email": getattr(u, "email", None),
        }
        for u in users
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
def list_user_groups() -> str:
    """List all user groups in the GoodData organization.

    Returns a JSON array of groups with their IDs and names.
    """
    sdk = _get_sdk()
    groups = sdk.catalog_user.list_user_groups()

    result = [{"id": g.id, "name": getattr(g, "name", None)} for g in groups]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_user_group_members(group_id: str) -> str:
    """Get all members of a specific user group.

    Args:
        group_id: The user group ID.

    Returns a JSON array of user IDs in the group.
    """
    sdk = _get_sdk()
    decl_users = sdk.catalog_user.get_declarative_users()

    members = []
    for u in decl_users.users:
        if u.user_groups:
            for ug in u.user_groups:
                if ug.id == group_id:
                    members.append(u.id)
                    break

    return json.dumps({"group_id": group_id, "members": members}, indent=2)


# =============================================================================
# QUERY TOOLS (Read-Only)
# =============================================================================


@mcp.tool()
def get_insight_metadata(insight_id: str, workspace_id: str | None = None) -> str:
    """Get detailed metadata for a specific insight/visualization.

    Returns metadata including tags, creation/modification dates, creator info,
    and related objects (metrics, attributes, datasets).

    Args:
        insight_id: The insight ID to get metadata for.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns metadata as JSON including:
        - id, title, description
        - tags (array of strings)
        - createdAt, modifiedAt (timestamps)
        - createdBy, modifiedBy (user info)
        - origin (originType, originId)
        - visualizationType (e.g., "table", "bar", "line")
        - filters (applied filters)
        - metrics (referenced metrics)
        - attributes (referenced attributes)
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    ws_id = _get_workspace_id(workspace_id)

    # Make direct API request to get full metadata
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    params = {"include": "createdBy,modifiedBy"}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    viz_data = data.get("data", {})
    attrs = viz_data.get("attributes", {})
    relationships = viz_data.get("relationships", {})
    meta = viz_data.get("meta", {})
    included = data.get("included", [])

    # Build user lookup from included data
    user_lookup = {}
    for item in included:
        if item.get("type") == "userIdentifier":
            user_attrs = item.get("attributes", {})
            user_lookup[item["id"]] = {
                "id": item["id"],
                "firstname": user_attrs.get("firstname"),
                "lastname": user_attrs.get("lastname"),
                "email": user_attrs.get("email"),
            }

    # Extract creator/modifier info
    created_by_id = relationships.get("createdBy", {}).get("data", {}).get("id")
    modified_by_id = relationships.get("modifiedBy", {}).get("data", {}).get("id")

    # Extract visualization type from content
    content = attrs.get("content", {})
    vis_url = content.get("visualizationUrl", "")
    vis_type = vis_url.replace("local:", "") if vis_url.startswith("local:") else vis_url

    # Extract referenced metrics and attributes from buckets
    metrics = []
    attributes = []
    for bucket in content.get("buckets", []):
        for item in bucket.get("items", []):
            if "measure" in item:
                measure = item["measure"]
                metric_id = (
                    measure.get("definition", {})
                    .get("measureDefinition", {})
                    .get("item", {})
                    .get("identifier", {})
                    .get("id")
                )
                if metric_id:
                    metrics.append(
                        {
                            "id": metric_id,
                            "title": measure.get("title"),
                        }
                    )
            if "attribute" in item:
                attr_item = item["attribute"]
                attr_id = attr_item.get("displayForm", {}).get("identifier", {}).get("id")
                if attr_id:
                    attributes.append({"id": attr_id})

    # Extract filters
    filters = []
    for f in content.get("filters", []):
        if "positiveAttributeFilter" in f:
            pf = f["positiveAttributeFilter"]
            filters.append(
                {
                    "type": "positive",
                    "attribute": pf.get("displayForm", {}).get("identifier", {}).get("id"),
                    "values": pf.get("in", {}).get("values", []),
                }
            )
        elif "negativeAttributeFilter" in f:
            nf = f["negativeAttributeFilter"]
            filters.append(
                {
                    "type": "negative",
                    "attribute": nf.get("displayForm", {}).get("identifier", {}).get("id"),
                    "values": nf.get("notIn", {}).get(
                        "values", nf.get("notIn", {}).get("uris", [])
                    ),
                }
            )

    result = {
        "id": viz_data.get("id"),
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "tags": attrs.get("tags", []),
        "createdAt": attrs.get("createdAt"),
        "modifiedAt": attrs.get("modifiedAt"),
        "createdBy": user_lookup.get(created_by_id) if created_by_id else None,
        "modifiedBy": user_lookup.get(modified_by_id) if modified_by_id else None,
        "origin": meta.get("origin"),
        "visualizationType": vis_type,
        "metrics": metrics,
        "attributes": attributes,
        "filters": filters,
        "areRelationsValid": attrs.get("areRelationsValid", attrs.get("are_relations_valid")),
    }

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_insight_data(insight_id: str, workspace_id: str | None = None) -> str:
    """Get data from a specific insight/visualization.

    Args:
        insight_id: The insight ID to query.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.

    Returns the insight data as JSON with metadata and rows.
    """
    _load_env()
    from gooddata_sdk import GoodDataSdk
    from gooddata_pandas import GoodPandas

    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    ws_id = _get_workspace_id(workspace_id)

    # Get visualization metadata
    sdk = GoodDataSdk.create(host, token)
    viz = sdk.visualizations.get_visualization(ws_id, insight_id)

    # Get data via GoodPandas
    gp = GoodPandas(host, token)
    df = gp.data_frames(ws_id).for_visualization(insight_id)

    result = {
        "id": viz.id,
        "title": viz.title,
        "description": viz.description,
        "columns": list(df.columns),
        "row_count": len(df),
        "data": df.to_dict(orient="records") if len(df) > 0 else [],
    }

    return json.dumps(result, indent=2, default=str)


# =============================================================================
# EXPORT TOOLS (Read-Only - exports to local files)
# =============================================================================


@mcp.tool()
def export_dashboard_pdf(
    dashboard_id: str,
    workspace_id: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a dashboard to PDF.

    Args:
        dashboard_id: The dashboard ID to export.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.
        output_path: Optional output file path. Defaults to ./exports/<dashboard_id>.pdf

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    if output_path is None:
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{dashboard_id}.pdf")

    sdk.export.export_pdf(
        workspace_id=ws_id,
        dashboard_id=dashboard_id,
        file_name=output_path,
    )

    return json.dumps(
        {
            "success": True,
            "path": os.path.abspath(output_path),
        }
    )


@mcp.tool()
def export_visualization_csv(
    visualization_id: str,
    workspace_id: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a visualization to CSV.

    Args:
        visualization_id: The visualization ID to export.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.
        output_path: Optional output file path. Defaults to ./exports/<visualization_id>.csv

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    if output_path is None:
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{visualization_id}.csv")

    sdk.export.export_tabular_by_visualization_id(
        workspace_id=ws_id,
        visualization_id=visualization_id,
        file_name=output_path,
        file_format="CSV",
    )

    return json.dumps(
        {
            "success": True,
            "path": os.path.abspath(output_path),
        }
    )


@mcp.tool()
def export_visualization_xlsx(
    visualization_id: str,
    workspace_id: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a visualization to Excel (XLSX).

    Args:
        visualization_id: The visualization ID to export.
        workspace_id: The workspace ID. Uses GOODDATA_WORKSPACE env var if not provided.
        output_path: Optional output file path. Defaults to ./exports/<visualization_id>.xlsx

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _get_workspace_id(workspace_id)

    if output_path is None:
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{visualization_id}.xlsx")

    sdk.export.export_tabular_by_visualization_id(
        workspace_id=ws_id,
        visualization_id=visualization_id,
        file_name=output_path,
        file_format="XLSX",
    )

    return json.dumps(
        {
            "success": True,
            "path": os.path.abspath(output_path),
        }
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
