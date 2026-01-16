#!/usr/bin/env python3
"""MCP Server for GoodData SDK CLI.

This exposes GoodData operations as MCP tools for use with Claude Code.

Read operations are available for all objects.
Write operations use a two-phase commit pattern (preview → apply) with
automatic backups and audit logging for safety.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("gooddata")

# Config file location
CONFIG_PATH = Path.home() / ".config" / "gooddata" / "workspaces.yaml"

# Stackless config directory for customer-specific backups and audit logs
STACKLESS_GOODDATA_DIR = Path.home() / ".config" / "stackless" / "gooddata"


def _get_backup_dir(customer: str) -> Path:
    """Get customer-specific backup directory."""
    backup_dir = STACKLESS_GOODDATA_DIR / customer / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _get_audit_log_path(customer: str) -> Path:
    """Get customer-specific audit log path."""
    log_dir = STACKLESS_GOODDATA_DIR / customer
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "audit.jsonl"


def _save_backup(customer: str, object_type: str, object_id: str, data: dict) -> Path:
    """Save a backup of an object before modification.

    Args:
        customer: Customer name (tpp, dlg, danceone).
        object_type: Type of object (e.g., 'visualizationObject').
        object_id: ID of the object.
        data: Full API response data to backup.

    Returns:
        Path to the backup file.
    """
    backup_dir = _get_backup_dir(customer)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use short object ID for filename
    short_id = object_id[:8] if len(object_id) > 8 else object_id
    backup_path = backup_dir / f"{object_type}_{short_id}_{timestamp}.json"

    backup_data = {
        "backed_up_at": datetime.now().isoformat(),
        "customer": customer,
        "object_type": object_type,
        "object_id": object_id,
        "data": data,
    }

    with open(backup_path, "w") as f:
        json.dump(backup_data, f, indent=2, default=str)

    return backup_path


def _log_audit(
    customer: str,
    operation: str,
    object_id: str,
    status: str,
    details: dict | None = None,
):
    """Append an entry to the customer's audit log.

    Args:
        customer: Customer name (tpp, dlg, danceone).
        operation: Name of the operation performed.
        object_id: ID of the affected object.
        status: Status of the operation ('success', 'error', 'preview').
        details: Optional additional details.
    """
    log_path = _get_audit_log_path(customer)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "object_id": object_id,
        "status": status,
        "details": details or {},
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _resolve_customer_name(customer: str | None = None) -> str:
    """Resolve customer name from parameter or CWD.

    Similar to _resolve_workspace_id but returns the customer name instead.
    """
    customers = _load_customer_config()
    available = ", ".join(customers.keys())

    if customer is not None:
        if customer not in customers:
            raise ValueError(f"Unknown customer '{customer}'. Available: {available}")
        return customer

    cwd = os.getcwd()
    for name, cust_config in customers.items():
        project_path = cust_config.get("project_path", "")
        if project_path and cwd.startswith(project_path):
            return name

    raise ValueError(
        f"Customer must be specified. Available: {available}. "
        f"Current directory ({cwd}) does not match any customer project."
    )


def _load_env():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _load_customer_config() -> dict:
    """Load customer configuration from workspaces.yaml."""
    if not CONFIG_PATH.exists():
        raise ValueError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    return config.get("customers", {})


def _resolve_workspace_id(customer: str | None = None) -> str:
    """Resolve workspace_id from customer name.

    Resolution order:
    1. Customer name → lookup in config
    2. Auto-detect from CWD via project_path
    3. Error with helpful message (list available customers)

    Args:
        customer: Customer name (tpp, dlg, danceone). Optional if CWD is inside a customer project.

    Returns:
        The workspace_id for the resolved customer.
    """
    customers = _load_customer_config()
    available = ", ".join(customers.keys())

    # 1. Explicit customer parameter
    if customer is not None:
        if customer not in customers:
            raise ValueError(f"Unknown customer '{customer}'. Available: {available}")
        return customers[customer]["workspace_id"]

    # 2. Auto-detect from current working directory
    cwd = os.getcwd()
    for name, cust_config in customers.items():
        project_path = cust_config.get("project_path", "")
        if project_path and cwd.startswith(project_path):
            return cust_config["workspace_id"]

    # 3. No match - require explicit customer
    raise ValueError(
        f"Customer must be specified. Available: {available}. "
        f"Current directory ({cwd}) does not match any customer project."
    )


def _get_sdk():
    """Get GoodData SDK instance."""
    _load_env()
    from gooddata_sdk import GoodDataSdk

    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    return GoodDataSdk.create(host, token)


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
def list_insights(customer: str | None = None) -> str:
    """List all insights (visualizations) in a workspace.

    Args:
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON array of insights with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    result = [{"id": viz.id, "title": viz.title} for viz in am.analytics.visualization_objects]
    return json.dumps(result, indent=2)


@mcp.tool()
def list_dashboards(customer: str | None = None) -> str:
    """List all dashboards in a workspace.

    Args:
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON array of dashboards with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    result = [{"id": db.id, "title": db.title} for db in am.analytics.analytical_dashboards]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_dashboard_filters(dashboard_id: str, customer: str | None = None) -> str:
    """Get all filters configured on a specific dashboard.

    Args:
        dashboard_id: The dashboard ID to get filters from.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON object with dashboard filter information including:
        - attribute filters (dropdown filters) with display form IDs
        - date filters with granularity and range
        - current filter values/selections
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    # Find the dashboard
    dashboard = None
    for db in am.analytics.analytical_dashboards:
        if db.id == dashboard_id:
            dashboard = db
            break

    if not dashboard:
        return json.dumps({"error": f"Dashboard '{dashboard_id}' not found"})

    content = dashboard.content

    # Extract filter context reference
    filter_context_ref = content.get("filterContextRef", {})
    filter_context_id = filter_context_ref.get("identifier", {}).get("id")

    # Look up the filterContext object to get the actual filters
    filter_context_content = None
    if filter_context_id:
        for fc in am.analytics.filter_contexts:
            if fc.id == filter_context_id:
                filter_context_content = fc.content
                break

    # Parse the filters from the filter context
    attribute_filters = []
    date_filters = []

    if filter_context_content:
        for f in filter_context_content.get("filters", []):
            if "attributeFilter" in f:
                af = f["attributeFilter"]
                # Handle both nested and flat identifier formats
                display_form = af.get("displayForm", {})
                identifier = display_form.get("identifier", display_form)
                if isinstance(identifier, dict):
                    display_form_id = identifier.get("id", identifier.get("identifier"))
                else:
                    display_form_id = identifier

                attribute_filters.append(
                    {
                        "displayForm": display_form_id,
                        "localIdentifier": af.get("localIdentifier"),
                        "negativeSelection": af.get("negativeSelection", False),
                        "selectionMode": af.get("selectionMode", "multi"),
                        "selectedValues": af.get("attributeElements", {}).get("uris", []),
                    }
                )

            elif "dateFilter" in f:
                df = f["dateFilter"]
                date_filters.append(
                    {
                        "type": df.get("type"),
                        "granularity": df.get("granularity"),
                        "from": df.get("from"),
                        "to": df.get("to"),
                        "localIdentifier": df.get("localIdentifier"),
                    }
                )

    result = {
        "dashboard_id": dashboard_id,
        "dashboard_title": dashboard.title,
        "filter_context_id": filter_context_id,
        "attribute_filters": attribute_filters,
        "attribute_filter_count": len(attribute_filters),
        "date_filters": date_filters,
        "date_filter_count": len(date_filters),
    }
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_dashboard_insights(dashboard_id: str, customer: str | None = None) -> str:
    """Get all insights (visualizations) contained in a specific dashboard.

    Args:
        dashboard_id: The dashboard ID to get insights from.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON object with dashboard info and an array of insights with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
def list_metrics(customer: str | None = None) -> str:
    """List all metrics in a workspace.

    Args:
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON array of metrics with all available properties.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
def list_datasets(customer: str | None = None) -> str:
    """List all datasets in a workspace.

    Args:
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns a JSON array of datasets with their IDs and titles.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)

    result = [{"id": ds.id, "title": ds.title} for ds in catalog.datasets]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_logical_data_model(
    customer: str | None = None,
    output_path: str | None = None,
) -> str:
    """Get the logical data model (LDM) for a workspace.

    The LDM contains all datasets, attributes, labels, facts, and their relationships.
    This is useful for understanding the data structure and for documentation.

    Args:
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        output_path: Optional file path to save the LDM. If provided, saves as JSON file.

    Returns:
        JSON containing the full logical data model structure, or path to saved file.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
def get_insight_metadata(insight_id: str, customer: str | None = None) -> str:
    """Get detailed metadata for a specific insight/visualization.

    Returns metadata including tags, creation/modification dates, creator info,
    and related objects (metrics, attributes, datasets).

    Args:
        insight_id: The insight ID to get metadata for.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

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

    ws_id = _resolve_workspace_id(customer)

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
def get_insight_data(insight_id: str, customer: str | None = None) -> str:
    """Get data from a specific insight/visualization.

    Args:
        insight_id: The insight ID to query.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns the insight data as JSON with metadata and rows.
    """
    _load_env()
    from gooddata_sdk import GoodDataSdk
    from gooddata_pandas import GoodPandas

    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    ws_id = _resolve_workspace_id(customer)

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
    customer: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a dashboard to PDF.

    Args:
        dashboard_id: The dashboard ID to export.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        output_path: Optional output file path. Defaults to ./exports/<dashboard_id>.pdf

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
    customer: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a visualization to CSV.

    Args:
        visualization_id: The visualization ID to export.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        output_path: Optional output file path. Defaults to ./exports/<visualization_id>.csv

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
    customer: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export a visualization to Excel (XLSX).

    Args:
        visualization_id: The visualization ID to export.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        output_path: Optional output file path. Defaults to ./exports/<visualization_id>.xlsx

    Returns the path to the exported file.
    """
    sdk = _get_sdk()
    ws_id = _resolve_workspace_id(customer)

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
# METRIC TOOLS (Read + Write)
# =============================================================================


@mcp.tool()
def get_metric(metric_id: str, customer: str | None = None) -> str:
    """Get detailed definition for a specific metric.

    Returns the full metric definition including MAQL, format, and metadata.

    Args:
        metric_id: The metric ID to get.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with metric details including:
        - id, title, description
        - format (e.g., "#,##0.00", "$#,##0.00", "#,##0.00%")
        - maql (the metric definition)
        - tags, createdAt, modifiedAt
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    ws_id = _resolve_workspace_id(customer)

    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    attrs = data["data"]["attributes"]
    content = attrs.get("content", {})

    result = {
        "id": data["data"]["id"],
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "format": content.get("format"),
        "maql": content.get("maql"),
        "tags": attrs.get("tags", []),
        "createdAt": attrs.get("createdAt"),
        "modifiedAt": attrs.get("modifiedAt"),
    }

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def preview_update_metric(
    metric_id: str,
    customer: str | None = None,
    title: str | None = None,
    description: str | None = None,
    format: str | None = None,
    maql: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Preview updating a metric (READ-ONLY).

    This shows what changes would be made to the metric.
    No changes are made. Use apply_update_metric to execute the change.

    Args:
        metric_id: The metric ID to update.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        title: New title for the metric. None to keep current.
        description: New description. None to keep current.
        format: New format string (e.g., "#,##0.00%", "$#,##0.00"). None to keep current.
        maql: New MAQL definition. None to keep current.
        tags: New tags list. None to keep current.

    Returns:
        JSON with:
        - current_values: Current metric properties
        - proposed_changes: What would be changed
        - confirmation_token: Token to pass to apply_update_metric
        - next_step: Instructions for applying the change
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current metric definition
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    attrs = data["data"]["attributes"]
    content = attrs.get("content", {})

    current_values = {
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "format": content.get("format"),
        "maql": content.get("maql"),
        "tags": attrs.get("tags", []),
    }

    # Determine what would change
    proposed_changes = {}
    if title is not None and title != current_values["title"]:
        proposed_changes["title"] = {"from": current_values["title"], "to": title}
    if description is not None and description != current_values["description"]:
        proposed_changes["description"] = {"from": current_values["description"], "to": description}
    if format is not None and format != current_values["format"]:
        proposed_changes["format"] = {"from": current_values["format"], "to": format}
    if maql is not None and maql != current_values["maql"]:
        proposed_changes["maql"] = {"from": current_values["maql"], "to": maql}
    if tags is not None and tags != current_values["tags"]:
        proposed_changes["tags"] = {"from": current_values["tags"], "to": tags}

    if not proposed_changes:
        return json.dumps(
            {
                "metric_id": metric_id,
                "message": "No changes proposed. All provided values match current values.",
                "current_values": current_values,
            },
            indent=2,
        )

    # Generate confirmation token
    token_data = f"{metric_id}:{json.dumps(proposed_changes, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_update_metric",
        object_id=metric_id,
        status="preview",
        details={"changes": list(proposed_changes.keys())},
    )

    result = {
        "metric_id": metric_id,
        "metric_title": current_values["title"],
        "current_values": current_values,
        "proposed_changes": proposed_changes,
        "change_count": len(proposed_changes),
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To apply this change, call: apply_update_metric("
            f"metric_id='{metric_id}', confirmation_token='{confirmation_token}', "
            + ", ".join(f"{k}='{v['to']}'" for k, v in proposed_changes.items() if k != "tags")
            + (f", tags={proposed_changes['tags']['to']}" if "tags" in proposed_changes else "")
            + f", customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_update_metric(
    metric_id: str,
    confirmation_token: str,
    customer: str | None = None,
    title: str | None = None,
    description: str | None = None,
    format: str | None = None,
    maql: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Apply updates to a metric (WRITE OPERATION).

    This modifies the metric in GoodData. A backup is automatically created
    before any changes are made.

    You must first call preview_update_metric to get the confirmation_token.

    Args:
        metric_id: The metric ID to update.
        confirmation_token: Token from preview_update_metric.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        title: New title for the metric. None to keep current.
        description: New description. None to keep current.
        format: New format string (e.g., "#,##0.00%", "$#,##0.00"). None to keep current.
        maql: New MAQL definition. None to keep current.
        tags: New tags list. None to keep current.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - backup_path: Path to the backup file (for rollback if needed)
        - changes_applied: What was changed
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current metric definition
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    attrs = data["data"]["attributes"]
    content = attrs.get("content", {})

    current_values = {
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "format": content.get("format"),
        "maql": content.get("maql"),
        "tags": attrs.get("tags", []),
    }

    # Determine what would change (must match preview)
    proposed_changes = {}
    if title is not None and title != current_values["title"]:
        proposed_changes["title"] = {"from": current_values["title"], "to": title}
    if description is not None and description != current_values["description"]:
        proposed_changes["description"] = {"from": current_values["description"], "to": description}
    if format is not None and format != current_values["format"]:
        proposed_changes["format"] = {"from": current_values["format"], "to": format}
    if maql is not None and maql != current_values["maql"]:
        proposed_changes["maql"] = {"from": current_values["maql"], "to": maql}
    if tags is not None and tags != current_values["tags"]:
        proposed_changes["tags"] = {"from": current_values["tags"], "to": tags}

    if not proposed_changes:
        return json.dumps(
            {
                "success": False,
                "error": "No changes to apply. All provided values match current values.",
            },
            indent=2,
        )

    # Verify confirmation token matches current state
    token_data = f"{metric_id}:{json.dumps(proposed_changes, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_update_metric",
            object_id=metric_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The metric may have changed since preview.",
                "message": "Please run preview_update_metric again to get a new token.",
            },
            indent=2,
        )

    # Save backup BEFORE making any changes
    backup_path = _save_backup(customer_name, "metric", metric_id, data)

    # Apply changes to the data structure
    if title is not None:
        attrs["title"] = title
    if description is not None:
        attrs["description"] = description
    if format is not None:
        content["format"] = format
    if maql is not None:
        content["maql"] = maql
    if tags is not None:
        attrs["tags"] = tags

    # Update the metric via PUT
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_update_metric",
            object_id=metric_id,
            status="error",
            details={"error": str(e), "backup_path": str(backup_path)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to update metric: {e}",
                "backup_path": str(backup_path),
                "message": "Backup was saved. Use restore_metric_from_backup to restore if needed.",
            },
            indent=2,
        )

    # Log successful change
    _log_audit(
        customer=customer_name,
        operation="apply_update_metric",
        object_id=metric_id,
        status="success",
        details={
            "changes": proposed_changes,
            "backup_path": str(backup_path),
        },
    )

    return json.dumps(
        {
            "success": True,
            "metric_id": metric_id,
            "backup_path": str(backup_path),
            "changes_applied": proposed_changes,
            "message": f"Successfully updated metric '{metric_id}'. Backup saved.",
        },
        indent=2,
    )


@mcp.tool()
def preview_create_metric(
    metric_id: str,
    title: str,
    maql: str,
    customer: str | None = None,
    description: str | None = None,
    format: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Preview creating a new metric (READ-ONLY).

    This shows what metric would be created.
    No changes are made. Use apply_create_metric to execute the creation.

    Args:
        metric_id: The ID for the new metric (lowercase, underscores, no spaces).
        title: Display title for the metric.
        maql: MAQL definition for the metric.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        description: Optional description.
        format: Optional format string (e.g., "#,##0.00%", "$#,##0.00").
        tags: Optional list of tags.

    Returns:
        JSON with:
        - metric_definition: The metric that would be created
        - confirmation_token: Token to pass to apply_create_metric
        - next_step: Instructions for applying the creation
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Check if metric already exists
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return json.dumps(
            {
                "success": False,
                "error": f"Metric '{metric_id}' already exists. Use preview_update_metric instead.",
            },
            indent=2,
        )

    # Build the metric definition
    metric_definition = {
        "id": metric_id,
        "title": title,
        "maql": maql,
        "description": description or "",
        "format": format or "#,##0",
        "tags": tags or [],
    }

    # Generate confirmation token
    token_data = f"create:{metric_id}:{json.dumps(metric_definition, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_create_metric",
        object_id=metric_id,
        status="preview",
        details={"title": title},
    )

    result = {
        "action": "create",
        "metric_definition": metric_definition,
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To create this metric, call: apply_create_metric("
            f"metric_id='{metric_id}', title='{title}', maql='{maql}', "
            f"confirmation_token='{confirmation_token}', customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_create_metric(
    metric_id: str,
    title: str,
    maql: str,
    confirmation_token: str,
    customer: str | None = None,
    description: str | None = None,
    format: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Create a new metric (WRITE OPERATION).

    You must first call preview_create_metric to get the confirmation_token.

    Args:
        metric_id: The ID for the new metric (lowercase, underscores, no spaces).
        title: Display title for the metric.
        maql: MAQL definition for the metric.
        confirmation_token: Token from preview_create_metric.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        description: Optional description.
        format: Optional format string (e.g., "#,##0.00%", "$#,##0.00").
        tags: Optional list of tags.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - metric_id: The ID of the created metric
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Build and verify the metric definition matches token
    metric_definition = {
        "id": metric_id,
        "title": title,
        "maql": maql,
        "description": description or "",
        "format": format or "#,##0",
        "tags": tags or [],
    }

    token_data = f"create:{metric_id}:{json.dumps(metric_definition, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_create_metric",
            object_id=metric_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. Parameters may have changed since preview.",
                "message": "Please run preview_create_metric again to get a new token.",
            },
            indent=2,
        )

    # Build the API payload
    payload = {
        "data": {
            "type": "metric",
            "id": metric_id,
            "attributes": {
                "title": title,
                "description": description or "",
                "tags": tags or [],
                "content": {
                    "maql": maql,
                    "format": format or "#,##0",
                },
            },
        }
    }

    # Create the metric via POST
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_create_metric",
            object_id=metric_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to create metric: {e}",
            },
            indent=2,
        )

    # Log successful creation
    _log_audit(
        customer=customer_name,
        operation="apply_create_metric",
        object_id=metric_id,
        status="success",
        details={"title": title, "maql": maql},
    )

    return json.dumps(
        {
            "success": True,
            "metric_id": metric_id,
            "title": title,
            "message": f"Successfully created metric '{metric_id}'.",
        },
        indent=2,
    )


@mcp.tool()
def preview_delete_metric(
    metric_id: str,
    customer: str | None = None,
) -> str:
    """Preview deleting a metric (READ-ONLY).

    This shows the metric that would be deleted and checks for dependencies.
    No changes are made. Use apply_delete_metric to execute the deletion.

    Args:
        metric_id: The metric ID to delete.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - metric_to_delete: The metric that would be deleted
        - confirmation_token: Token to pass to apply_delete_metric
        - next_step: Instructions for applying the deletion
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current metric definition
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Metric '{metric_id}' not found.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    attrs = data["data"]["attributes"]
    content = attrs.get("content", {})

    metric_to_delete = {
        "id": metric_id,
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "format": content.get("format"),
        "maql": content.get("maql"),
        "tags": attrs.get("tags", []),
    }

    # Generate confirmation token
    token_data = f"delete:{metric_id}:{attrs.get('title')}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_delete_metric",
        object_id=metric_id,
        status="preview",
        details={"title": attrs.get("title")},
    )

    result = {
        "action": "delete",
        "metric_to_delete": metric_to_delete,
        "warning": "This will permanently delete the metric. A backup will be saved before deletion.",
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To delete this metric, call: apply_delete_metric("
            f"metric_id='{metric_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_delete_metric(
    metric_id: str,
    confirmation_token: str,
    customer: str | None = None,
) -> str:
    """Delete a metric (WRITE OPERATION).

    This permanently deletes the metric from GoodData. A backup is automatically
    created before deletion for potential recovery.

    You must first call preview_delete_metric to get the confirmation_token.

    Args:
        metric_id: The metric ID to delete.
        confirmation_token: Token from preview_delete_metric.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - backup_path: Path to the backup file (for recovery if needed)
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current metric to verify and backup
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{metric_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Metric '{metric_id}' not found.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    attrs = data["data"]["attributes"]

    # Verify confirmation token
    token_data = f"delete:{metric_id}:{attrs.get('title')}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_metric",
            object_id=metric_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The metric may have changed since preview.",
                "message": "Please run preview_delete_metric again to get a new token.",
            },
            indent=2,
        )

    # Save backup BEFORE deletion
    backup_path = _save_backup(customer_name, "metric", metric_id, data)

    # Delete the metric
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_metric",
            object_id=metric_id,
            status="error",
            details={"error": str(e), "backup_path": str(backup_path)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to delete metric: {e}",
                "backup_path": str(backup_path),
            },
            indent=2,
        )

    # Log successful deletion
    _log_audit(
        customer=customer_name,
        operation="apply_delete_metric",
        object_id=metric_id,
        status="success",
        details={
            "title": attrs.get("title"),
            "backup_path": str(backup_path),
        },
    )

    return json.dumps(
        {
            "success": True,
            "metric_id": metric_id,
            "title": attrs.get("title"),
            "backup_path": str(backup_path),
            "message": f"Successfully deleted metric '{metric_id}'. Backup saved for recovery.",
        },
        indent=2,
    )


@mcp.tool()
def restore_metric_from_backup(
    backup_path: str,
    customer: str | None = None,
) -> str:
    """Restore a metric from a backup file (WRITE OPERATION).

    Use this to recover a deleted metric or undo changes.
    The backup_path is provided in the response of apply_* operations.

    Args:
        backup_path: Path to the backup file (from a previous write operation).
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with success status and details.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Load backup file
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return json.dumps(
            {
                "success": False,
                "error": f"Backup file not found: {backup_path}",
            },
            indent=2,
        )

    with open(backup_file) as f:
        backup = json.load(f)

    object_type = backup.get("object_type")
    object_id = backup.get("object_id")
    data = backup.get("data")
    backed_up_at = backup.get("backed_up_at")

    if object_type != "metric":
        return json.dumps(
            {
                "success": False,
                "error": f"This function only restores metrics. Got: {object_type}",
                "message": "Use restore_insight_from_backup for visualization objects.",
            },
            indent=2,
        )

    # Check if metric exists (update) or not (create)
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics/{object_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    check_response = requests.get(url, headers=headers)
    metric_exists = check_response.status_code == 200

    try:
        if metric_exists:
            # Update existing metric
            response = requests.put(url, headers=headers, json=data)
        else:
            # Create metric (was deleted)
            create_url = f"{host}/api/v1/entities/workspaces/{ws_id}/metrics"
            response = requests.post(create_url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="restore_metric_from_backup",
            object_id=object_id,
            status="error",
            details={"error": str(e), "backup_path": backup_path},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to restore metric: {e}",
            },
            indent=2,
        )

    # Log successful restore
    _log_audit(
        customer=customer_name,
        operation="restore_metric_from_backup",
        object_id=object_id,
        status="success",
        details={
            "restored_from": backup_path,
            "original_backup_time": backed_up_at,
            "action": "updated" if metric_exists else "created",
        },
    )

    return json.dumps(
        {
            "success": True,
            "metric_id": object_id,
            "action": "updated" if metric_exists else "recreated",
            "restored_from": backup_path,
            "original_backup_time": backed_up_at,
            "message": "Successfully restored metric from backup.",
        },
        indent=2,
    )


# =============================================================================
# WRITE TOOLS (Two-Phase Commit: Preview → Apply)
# =============================================================================


@mcp.tool()
def preview_remove_duplicate_metrics(
    insight_id: str,
    customer: str | None = None,
) -> str:
    """Preview removing duplicate metrics from an insight (READ-ONLY).

    This analyzes the insight and shows which duplicate metrics would be removed.
    No changes are made. Use apply_remove_duplicate_metrics to execute the change.

    Args:
        insight_id: The insight ID to check for duplicate metrics.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - current_metrics: List of all metrics with their local identifiers
        - duplicates_found: List of duplicate metrics that would be removed
        - confirmation_token: Token to pass to apply_remove_duplicate_metrics
        - next_step: Instructions for applying the change
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current insight definition
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    title = data["data"]["attributes"].get("title", "")
    content = data["data"]["attributes"]["content"]
    buckets = content.get("buckets", [])

    # Find metrics bucket and identify duplicates
    current_metrics = []
    duplicates = []
    seen_metric_ids = {}

    for bucket in buckets:
        if bucket.get("localIdentifier") == "measures":
            for item in bucket.get("items", []):
                if "measure" in item:
                    measure = item["measure"]
                    metric_def = measure.get("definition", {}).get("measureDefinition", {})
                    metric_id = metric_def.get("item", {}).get("identifier", {}).get("id")
                    local_id = measure.get("localIdentifier")
                    metric_title = measure.get("title")

                    current_metrics.append(
                        {
                            "local_identifier": local_id,
                            "metric_id": metric_id,
                            "title": metric_title,
                        }
                    )

                    if metric_id in seen_metric_ids:
                        duplicates.append(
                            {
                                "local_identifier": local_id,
                                "metric_id": metric_id,
                                "title": metric_title,
                                "duplicate_of": seen_metric_ids[metric_id],
                            }
                        )
                    else:
                        seen_metric_ids[metric_id] = local_id

    # Generate confirmation token (hash of insight_id + duplicates)
    token_data = f"{insight_id}:{json.dumps(duplicates, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_remove_duplicate_metrics",
        object_id=insight_id,
        status="preview",
        details={"duplicates_count": len(duplicates)},
    )

    result = {
        "insight_id": insight_id,
        "insight_title": title,
        "current_metric_count": len(current_metrics),
        "current_metrics": current_metrics,
        "duplicates_found": duplicates,
        "duplicates_count": len(duplicates),
        "metrics_after_count": len(current_metrics) - len(duplicates),
        "confirmation_token": confirmation_token,
    }

    if duplicates:
        result["next_step"] = (
            f"To apply this change, call: apply_remove_duplicate_metrics("
            f"insight_id='{insight_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}')"
        )
    else:
        result["message"] = "No duplicate metrics found. No action needed."

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_remove_duplicate_metrics(
    insight_id: str,
    confirmation_token: str,
    customer: str | None = None,
) -> str:
    """Apply removal of duplicate metrics from an insight (WRITE OPERATION).

    This modifies the insight in GoodData. A backup is automatically created
    before any changes are made.

    You must first call preview_remove_duplicate_metrics to get the
    confirmation_token. This ensures you've reviewed what will be changed.

    Args:
        insight_id: The insight ID to modify.
        confirmation_token: Token from preview_remove_duplicate_metrics.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - backup_path: Path to the backup file (for rollback if needed)
        - removed_duplicates: List of removed duplicate metrics
        - new_metric_count: Number of metrics after removal
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current insight definition
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    content = data["data"]["attributes"]["content"]
    buckets = content.get("buckets", [])

    # Re-identify duplicates
    duplicates = []
    seen_metric_ids = {}

    for bucket in buckets:
        if bucket.get("localIdentifier") == "measures":
            for item in bucket.get("items", []):
                if "measure" in item:
                    measure = item["measure"]
                    metric_def = measure.get("definition", {}).get("measureDefinition", {})
                    metric_id = metric_def.get("item", {}).get("identifier", {}).get("id")
                    local_id = measure.get("localIdentifier")
                    metric_title = measure.get("title")

                    if metric_id in seen_metric_ids:
                        duplicates.append(
                            {
                                "local_identifier": local_id,
                                "metric_id": metric_id,
                                "title": metric_title,
                                "duplicate_of": seen_metric_ids[metric_id],
                            }
                        )
                    else:
                        seen_metric_ids[metric_id] = local_id

    # Verify confirmation token matches current state
    token_data = f"{insight_id}:{json.dumps(duplicates, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_remove_duplicate_metrics",
            object_id=insight_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The insight may have changed since preview.",
                "message": "Please run preview_remove_duplicate_metrics again to get a new token.",
            },
            indent=2,
        )

    if not duplicates:
        return json.dumps(
            {
                "success": False,
                "error": "No duplicate metrics found to remove.",
            },
            indent=2,
        )

    # Save backup BEFORE making any changes
    backup_path = _save_backup(customer_name, "visualizationObject", insight_id, data)

    # Remove duplicates from the measures bucket
    duplicate_local_ids = {d["local_identifier"] for d in duplicates}

    for bucket in buckets:
        if bucket.get("localIdentifier") == "measures":
            bucket["items"] = [
                item
                for item in bucket.get("items", [])
                if item.get("measure", {}).get("localIdentifier") not in duplicate_local_ids
            ]

    # Update the insight via PUT
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_remove_duplicate_metrics",
            object_id=insight_id,
            status="error",
            details={"error": str(e), "backup_path": str(backup_path)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to update insight: {e}",
                "backup_path": str(backup_path),
                "message": "Backup was saved. Use restore_insight_from_backup to restore if needed.",
            },
            indent=2,
        )

    # Log successful change
    _log_audit(
        customer=customer_name,
        operation="apply_remove_duplicate_metrics",
        object_id=insight_id,
        status="success",
        details={
            "removed_count": len(duplicates),
            "removed": duplicates,
            "backup_path": str(backup_path),
        },
    )

    return json.dumps(
        {
            "success": True,
            "insight_id": insight_id,
            "backup_path": str(backup_path),
            "removed_duplicates": duplicates,
            "removed_count": len(duplicates),
            "new_metric_count": len(seen_metric_ids),
            "message": f"Successfully removed {len(duplicates)} duplicate metric(s). Backup saved.",
        },
        indent=2,
    )


@mcp.tool()
def restore_insight_from_backup(
    backup_path: str,
    customer: str | None = None,
) -> str:
    """Restore an insight from a backup file (WRITE OPERATION).

    Use this to undo changes made by write operations. The backup_path
    is provided in the response of apply_* operations.

    Args:
        backup_path: Path to the backup file (from a previous write operation).
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with success status and details.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Load backup file
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return json.dumps(
            {
                "success": False,
                "error": f"Backup file not found: {backup_path}",
            },
            indent=2,
        )

    with open(backup_file) as f:
        backup = json.load(f)

    object_type = backup.get("object_type")
    object_id = backup.get("object_id")
    data = backup.get("data")
    backed_up_at = backup.get("backed_up_at")

    if object_type != "visualizationObject":
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported object type for restore: {object_type}",
                "message": "Currently only visualizationObject restores are supported.",
            },
            indent=2,
        )

    # Restore the object via PUT
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{object_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="restore_insight_from_backup",
            object_id=object_id,
            status="error",
            details={"error": str(e), "backup_path": backup_path},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to restore insight: {e}",
            },
            indent=2,
        )

    # Log successful restore
    _log_audit(
        customer=customer_name,
        operation="restore_insight_from_backup",
        object_id=object_id,
        status="success",
        details={
            "restored_from": backup_path,
            "original_backup_time": backed_up_at,
        },
    )

    return json.dumps(
        {
            "success": True,
            "object_id": object_id,
            "object_type": object_type,
            "restored_from": backup_path,
            "original_backup_time": backed_up_at,
            "message": f"Successfully restored {object_type} from backup.",
        },
        indent=2,
    )


# =============================================================================
# INSIGHT CRUD OPERATIONS
# =============================================================================

# Supported visualization types
VISUALIZATION_TYPES = {
    "table": "local:table",
    "bar": "local:bar",
    "column": "local:column",
    "line": "local:line",
    "area": "local:area",
    "pie": "local:pie",
    "donut": "local:donut",
    "headline": "local:headline",
    "scatter": "local:scatter",
    "bubble": "local:bubble",
    "heatmap": "local:heatmap",
    "treemap": "local:treemap",
    "combo": "local:combo",
    "combo2": "local:combo2",
    "bullet": "local:bullet",
    "geo": "local:pushpin",
    "funnel": "local:funnel",
    "pyramid": "local:pyramid",
    "sankey": "local:sankey",
    "dependencywheel": "local:dependencywheel",
    "waterfall": "local:waterfall",
    "repeater": "local:repeater",
}


@mcp.tool()
def list_visualization_types() -> str:
    """List all supported visualization types for insights.

    Returns a JSON object mapping simple type names to their GoodData visualizationUrl values.
    """
    return json.dumps(
        {
            "visualization_types": VISUALIZATION_TYPES,
            "usage": "Use the simple name (e.g., 'table', 'bar') when creating insights.",
        },
        indent=2,
    )


def _validate_metrics_exist(ws_id: str, metric_ids: list[str], sdk) -> tuple[bool, list[str]]:
    """Validate that all metric IDs exist in the workspace.

    Returns:
        Tuple of (all_valid, missing_ids)
    """
    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)
    existing_metrics = {m.id for m in catalog.metrics}
    missing = [m for m in metric_ids if m not in existing_metrics]
    return len(missing) == 0, missing


def _validate_labels_exist(ws_id: str, label_ids: list[str], sdk) -> tuple[bool, list[str]]:
    """Validate that all label IDs exist in the workspace.

    Returns:
        Tuple of (all_valid, missing_ids)
    """
    # Get all labels from datasets
    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)
    existing_labels = set()
    for dataset in catalog.datasets:
        for attr in dataset.attributes:
            for label in attr.labels:
                existing_labels.add(label.id)
    missing = [label_id for label_id in label_ids if label_id not in existing_labels]
    return len(missing) == 0, missing


def _validate_insights_exist(ws_id: str, insight_ids: list[str], sdk) -> tuple[bool, list[str]]:
    """Validate that all insight IDs exist in the workspace.

    Returns:
        Tuple of (all_valid, missing_ids)
    """
    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)
    existing_insights = {viz.id for viz in am.analytics.visualization_objects}
    missing = [i for i in insight_ids if i not in existing_insights]
    return len(missing) == 0, missing


def _build_dashboard_layout(
    insight_ids: list[str],
    columns: int = 2,
    section_title: str | None = None,
) -> dict:
    """Build a simple dashboard layout from insight IDs.

    Creates a single section with insights arranged in a grid.

    Args:
        insight_ids: List of insight IDs to include.
        columns: Number of columns (1-4).
        section_title: Optional section header.

    Returns:
        Dashboard content dict ready for API.
    """
    # Calculate grid width per column (total grid is 12 units)
    grid_width = 12 // columns

    # Build items
    items = []
    for insight_id in insight_ids:
        item = {
            "type": "IDashboardLayoutItem",
            "widget": {
                "type": "insight",
                "insight": {"identifier": {"id": insight_id, "type": "visualizationObject"}},
                "ignoreDashboardFilters": [],
                "drills": [],
                "title": "",  # Uses insight's title
                "description": "",
            },
            "size": {"xl": {"gridWidth": grid_width, "gridHeight": 22}},  # Default height
        }
        items.append(item)

    # Build section
    section = {
        "type": "IDashboardLayoutSection",
        "header": {"title": section_title} if section_title else {},
        "items": items,
    }

    # Build layout
    layout = {"type": "IDashboardLayout", "sections": [section]}

    return {"layout": layout, "version": "2"}


def _get_dashboard_by_id(host: str, token: str, ws_id: str, dashboard_id: str) -> dict | None:
    """Fetch dashboard by ID using direct API call.

    Args:
        host: GoodData host URL.
        token: API token.
        ws_id: Workspace ID.
        dashboard_id: Dashboard ID.

    Returns:
        Full API response data, or None if not found.
    """
    import requests

    url = f"{host}/api/v1/entities/workspaces/{ws_id}/analyticalDashboards/{dashboard_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _build_insight_content(
    visualization_type: str,
    metric_ids: list[str],
    attribute_ids: list[str] | None = None,
    filters: list[dict] | None = None,
) -> dict:
    """Build the insight content structure.

    Args:
        visualization_type: Simple type name (e.g., 'table', 'bar')
        metric_ids: List of metric IDs to include
        attribute_ids: Optional list of label IDs for grouping/rows
        filters: Optional list of filter definitions

    Returns:
        The content dict for the visualization object
    """
    import uuid

    # Build measures bucket
    measures_items = []
    for metric_id in metric_ids:
        measures_items.append(
            {
                "measure": {
                    "localIdentifier": uuid.uuid4().hex[:32],
                    "definition": {
                        "measureDefinition": {
                            "item": {"identifier": {"id": metric_id, "type": "metric"}},
                            "filters": [],
                        }
                    },
                    "title": metric_id,  # Will be replaced by actual title
                }
            }
        )

    buckets = [{"localIdentifier": "measures", "items": measures_items}]

    # Build attribute bucket if provided
    if attribute_ids:
        attribute_items = []
        for label_id in attribute_ids:
            attribute_items.append(
                {
                    "attribute": {
                        "localIdentifier": uuid.uuid4().hex[:32],
                        "displayForm": {"identifier": {"id": label_id, "type": "label"}},
                    }
                }
            )
        buckets.append({"localIdentifier": "attribute", "items": attribute_items})

    # Get the visualization URL
    viz_url = VISUALIZATION_TYPES.get(visualization_type.lower(), "local:table")

    return {
        "buckets": buckets,
        "filters": filters or [],
        "sorts": [],
        "properties": {},
        "visualizationUrl": viz_url,
        "version": "2",
    }


@mcp.tool()
def preview_create_insight(
    insight_id: str,
    title: str,
    visualization_type: str,
    metric_ids: list[str],
    customer: str | None = None,
    attribute_ids: list[str] | None = None,
    description: str | None = None,
    filters: list[dict] | None = None,
) -> str:
    """Preview creating a new insight/visualization (READ-ONLY).

    This validates the insight definition and shows what would be created.
    No changes are made. Use apply_create_insight to execute the creation.

    Args:
        insight_id: The ID for the new insight (lowercase, underscores, no spaces).
        title: Display title for the insight.
        visualization_type: Type of visualization (table, bar, line, pie, etc.).
            Use list_visualization_types() to see all options.
        metric_ids: List of metric IDs to include in the insight.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        attribute_ids: Optional list of label IDs for grouping/rows.
        description: Optional description.
        filters: Optional list of filter definitions.

    Returns:
        JSON with:
        - insight_definition: The insight that would be created
        - confirmation_token: Token to pass to apply_create_insight
        - next_step: Instructions for applying the creation
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)
    sdk = _get_sdk()

    # Validate visualization type
    if visualization_type.lower() not in VISUALIZATION_TYPES:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid visualization type: '{visualization_type}'",
                "valid_types": list(VISUALIZATION_TYPES.keys()),
            },
            indent=2,
        )

    # Check if insight already exists
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return json.dumps(
            {
                "success": False,
                "error": f"Insight '{insight_id}' already exists. Use preview_update_insight instead.",
            },
            indent=2,
        )

    # Validate metrics exist
    metrics_valid, missing_metrics = _validate_metrics_exist(ws_id, metric_ids, sdk)
    if not metrics_valid:
        return json.dumps(
            {
                "success": False,
                "error": "Some metrics do not exist in the workspace.",
                "missing_metrics": missing_metrics,
            },
            indent=2,
        )

    # Validate attributes/labels exist
    if attribute_ids:
        labels_valid, missing_labels = _validate_labels_exist(ws_id, attribute_ids, sdk)
        if not labels_valid:
            return json.dumps(
                {
                    "success": False,
                    "error": "Some labels/attributes do not exist in the workspace.",
                    "missing_labels": missing_labels,
                },
                indent=2,
            )

    # Build the insight definition for preview
    content = _build_insight_content(visualization_type, metric_ids, attribute_ids, filters)

    insight_definition = {
        "id": insight_id,
        "title": title,
        "description": description or "",
        "visualization_type": visualization_type,
        "metric_ids": metric_ids,
        "attribute_ids": attribute_ids or [],
        "filters": filters or [],
    }

    # Generate confirmation token
    token_data = f"create_insight:{insight_id}:{json.dumps(insight_definition, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_create_insight",
        object_id=insight_id,
        status="preview",
        details={"title": title, "visualization_type": visualization_type},
    )

    result = {
        "success": True,
        "action": "create_insight",
        "insight_definition": insight_definition,
        "content_preview": content,
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To create this insight, call: apply_create_insight("
            f"insight_id='{insight_id}', title='{title}', "
            f"visualization_type='{visualization_type}', "
            f"metric_ids={metric_ids}, "
            f"confirmation_token='{confirmation_token}', customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_create_insight(
    insight_id: str,
    title: str,
    visualization_type: str,
    metric_ids: list[str],
    confirmation_token: str,
    customer: str | None = None,
    attribute_ids: list[str] | None = None,
    description: str | None = None,
    filters: list[dict] | None = None,
) -> str:
    """Create a new insight/visualization (WRITE OPERATION).

    You must first call preview_create_insight to get the confirmation_token.

    Args:
        insight_id: The ID for the new insight (lowercase, underscores, no spaces).
        title: Display title for the insight.
        visualization_type: Type of visualization (table, bar, line, pie, etc.).
        metric_ids: List of metric IDs to include in the insight.
        confirmation_token: Token from preview_create_insight.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        attribute_ids: Optional list of label IDs for grouping/rows.
        description: Optional description.
        filters: Optional list of filter definitions.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - insight_id: The ID of the created insight
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Build and verify the insight definition matches token
    insight_definition = {
        "id": insight_id,
        "title": title,
        "description": description or "",
        "visualization_type": visualization_type,
        "metric_ids": metric_ids,
        "attribute_ids": attribute_ids or [],
        "filters": filters or [],
    }

    token_data = f"create_insight:{insight_id}:{json.dumps(insight_definition, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_create_insight",
            object_id=insight_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. Parameters may have changed since preview.",
                "message": "Please run preview_create_insight again to get a new token.",
            },
            indent=2,
        )

    # Build the content
    content = _build_insight_content(visualization_type, metric_ids, attribute_ids, filters)

    # Build the API payload
    payload = {
        "data": {
            "type": "visualizationObject",
            "id": insight_id,
            "attributes": {
                "title": title,
                "description": description or "",
                "content": content,
            },
        }
    }

    # Create the insight via POST
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_create_insight",
            object_id=insight_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to create insight: {e}",
            },
            indent=2,
        )

    # Log successful creation
    _log_audit(
        customer=customer_name,
        operation="apply_create_insight",
        object_id=insight_id,
        status="success",
        details={"title": title, "visualization_type": visualization_type},
    )

    return json.dumps(
        {
            "success": True,
            "insight_id": insight_id,
            "title": title,
            "visualization_type": visualization_type,
            "message": f"Successfully created insight '{title}'.",
        },
        indent=2,
    )


@mcp.tool()
def preview_update_insight(
    insight_id: str,
    customer: str | None = None,
    title: str | None = None,
    metric_ids: list[str] | None = None,
    attribute_ids: list[str] | None = None,
    description: str | None = None,
    visualization_type: str | None = None,
) -> str:
    """Preview updating an existing insight (READ-ONLY).

    This shows what changes would be made. A backup is created during preview.
    No changes are made. Use apply_update_insight to execute the update.

    Args:
        insight_id: The ID of the insight to update.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        title: New title (optional).
        metric_ids: New list of metric IDs (replaces existing).
        attribute_ids: New list of label IDs (replaces existing).
        description: New description.
        visualization_type: New visualization type.

    Returns:
        JSON with:
        - current: Current insight state
        - changes: What would change
        - confirmation_token: Token to pass to apply_update_insight
        - backup_path: Path to the backup file
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)
    sdk = _get_sdk()

    # Fetch current insight
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Insight '{insight_id}' not found. Use preview_create_insight to create a new one.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    # Create backup
    backup_path = _save_backup(customer_name, "visualizationObject", insight_id, data)

    current_attrs = data["data"]["attributes"]
    current_content = current_attrs.get("content", {})

    # Extract current state
    current_state = {
        "title": current_attrs.get("title", ""),
        "description": current_attrs.get("description", ""),
        "visualization_type": current_content.get("visualizationUrl", "").replace("local:", ""),
    }

    # Build changes dict
    changes = {}
    if title is not None and title != current_state["title"]:
        changes["title"] = {"from": current_state["title"], "to": title}
    if description is not None and description != current_state["description"]:
        changes["description"] = {"from": current_state["description"], "to": description}
    if visualization_type is not None:
        current_viz = current_state["visualization_type"]
        if visualization_type.lower() != current_viz:
            if visualization_type.lower() not in VISUALIZATION_TYPES:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Invalid visualization type: '{visualization_type}'",
                        "valid_types": list(VISUALIZATION_TYPES.keys()),
                    },
                    indent=2,
                )
            changes["visualization_type"] = {"from": current_viz, "to": visualization_type}

    # Validate new metrics if provided
    if metric_ids is not None:
        metrics_valid, missing_metrics = _validate_metrics_exist(ws_id, metric_ids, sdk)
        if not metrics_valid:
            return json.dumps(
                {
                    "success": False,
                    "error": "Some metrics do not exist in the workspace.",
                    "missing_metrics": missing_metrics,
                },
                indent=2,
            )
        changes["metric_ids"] = {"to": metric_ids}

    # Validate new attributes if provided
    if attribute_ids is not None:
        labels_valid, missing_labels = _validate_labels_exist(ws_id, attribute_ids, sdk)
        if not labels_valid:
            return json.dumps(
                {
                    "success": False,
                    "error": "Some labels/attributes do not exist in the workspace.",
                    "missing_labels": missing_labels,
                },
                indent=2,
            )
        changes["attribute_ids"] = {"to": attribute_ids}

    if not changes:
        return json.dumps(
            {
                "success": True,
                "message": "No changes specified.",
                "current": current_state,
            },
            indent=2,
        )

    # Generate confirmation token
    update_def = {
        "insight_id": insight_id,
        "changes": changes,
    }
    token_data = f"update_insight:{insight_id}:{json.dumps(update_def, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_update_insight",
        object_id=insight_id,
        status="preview",
        details={"changes": list(changes.keys())},
    )

    result = {
        "success": True,
        "action": "update_insight",
        "insight_id": insight_id,
        "current": current_state,
        "changes": changes,
        "backup_path": str(backup_path),
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To apply these changes, call: apply_update_insight("
            f"insight_id='{insight_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}', ...changed_params...)"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_update_insight(
    insight_id: str,
    confirmation_token: str,
    customer: str | None = None,
    title: str | None = None,
    metric_ids: list[str] | None = None,
    attribute_ids: list[str] | None = None,
    description: str | None = None,
    visualization_type: str | None = None,
) -> str:
    """Update an existing insight (WRITE OPERATION).

    You must first call preview_update_insight to get the confirmation_token.
    A backup was already created during preview.

    Args:
        insight_id: The ID of the insight to update.
        confirmation_token: Token from preview_update_insight.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.
        title: New title (optional).
        metric_ids: New list of metric IDs (replaces existing).
        attribute_ids: New list of label IDs (replaces existing).
        description: New description.
        visualization_type: New visualization type.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - insight_id: The ID of the updated insight
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current insight
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Insight '{insight_id}' not found.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    current_attrs = data["data"]["attributes"]
    current_content = current_attrs.get("content", {})

    # Build changes for token verification
    current_state = {
        "title": current_attrs.get("title", ""),
        "description": current_attrs.get("description", ""),
        "visualization_type": current_content.get("visualizationUrl", "").replace("local:", ""),
    }

    changes = {}
    if title is not None and title != current_state["title"]:
        changes["title"] = {"from": current_state["title"], "to": title}
    if description is not None and description != current_state["description"]:
        changes["description"] = {"from": current_state["description"], "to": description}
    if visualization_type is not None:
        current_viz = current_state["visualization_type"]
        if visualization_type.lower() != current_viz:
            changes["visualization_type"] = {"from": current_viz, "to": visualization_type}
    if metric_ids is not None:
        changes["metric_ids"] = {"to": metric_ids}
    if attribute_ids is not None:
        changes["attribute_ids"] = {"to": attribute_ids}

    # Verify token
    update_def = {
        "insight_id": insight_id,
        "changes": changes,
    }
    token_data = f"update_insight:{insight_id}:{json.dumps(update_def, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_update_insight",
            object_id=insight_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The insight may have changed since preview.",
                "message": "Please run preview_update_insight again to get a new token.",
            },
            indent=2,
        )

    # Apply changes to the data
    if title is not None:
        current_attrs["title"] = title
    if description is not None:
        current_attrs["description"] = description

    # Rebuild content if metrics, attributes, or viz type changed
    if metric_ids is not None or attribute_ids is not None or visualization_type is not None:
        new_viz_type = visualization_type or current_state["visualization_type"]
        # Use new values or extract from current
        new_metric_ids = metric_ids
        new_attribute_ids = attribute_ids

        if new_metric_ids is None:
            # Extract current metric IDs
            new_metric_ids = []
            for bucket in current_content.get("buckets", []):
                if bucket.get("localIdentifier") == "measures":
                    for item in bucket.get("items", []):
                        if "measure" in item:
                            metric_def = (
                                item["measure"].get("definition", {}).get("measureDefinition", {})
                            )
                            metric_id = metric_def.get("item", {}).get("identifier", {}).get("id")
                            if metric_id:
                                new_metric_ids.append(metric_id)

        if new_attribute_ids is None:
            # Extract current attribute IDs
            new_attribute_ids = []
            for bucket in current_content.get("buckets", []):
                if bucket.get("localIdentifier") == "attribute":
                    for item in bucket.get("items", []):
                        if "attribute" in item:
                            label_id = (
                                item["attribute"]
                                .get("displayForm", {})
                                .get("identifier", {})
                                .get("id")
                            )
                            if label_id:
                                new_attribute_ids.append(label_id)

        current_attrs["content"] = _build_insight_content(
            new_viz_type, new_metric_ids, new_attribute_ids, current_content.get("filters")
        )

    # Update via PUT
    headers["Content-Type"] = "application/vnd.gooddata.api+json"
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_update_insight",
            object_id=insight_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to update insight: {e}",
                "message": "Backup was saved during preview. Use restore_insight_from_backup to restore if needed.",
            },
            indent=2,
        )

    # Log successful update
    _log_audit(
        customer=customer_name,
        operation="apply_update_insight",
        object_id=insight_id,
        status="success",
        details={"changes": list(changes.keys())},
    )

    return json.dumps(
        {
            "success": True,
            "insight_id": insight_id,
            "changes_applied": list(changes.keys()),
            "message": f"Successfully updated insight '{insight_id}'.",
        },
        indent=2,
    )


@mcp.tool()
def preview_delete_insight(
    insight_id: str,
    customer: str | None = None,
) -> str:
    """Preview deleting an insight (READ-ONLY).

    This creates a backup and shows what would be deleted.
    No changes are made. Use apply_delete_insight to execute the deletion.

    Args:
        insight_id: The ID of the insight to delete.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - insight_to_delete: The insight that would be deleted
        - confirmation_token: Token to pass to apply_delete_insight
        - backup_path: Path to the backup file
        - warning: Deletion warning message
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current insight
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Insight '{insight_id}' not found.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    # Create backup
    backup_path = _save_backup(customer_name, "visualizationObject", insight_id, data)

    current_attrs = data["data"]["attributes"]
    title = current_attrs.get("title", "")

    # Generate confirmation token
    token_data = f"delete_insight:{insight_id}:{title}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview action
    _log_audit(
        customer=customer_name,
        operation="preview_delete_insight",
        object_id=insight_id,
        status="preview",
        details={"title": title, "backup_path": str(backup_path)},
    )

    result = {
        "success": True,
        "action": "delete_insight",
        "insight_to_delete": {
            "id": insight_id,
            "title": title,
        },
        "backup_path": str(backup_path),
        "confirmation_token": confirmation_token,
        "warning": "THIS WILL PERMANENTLY DELETE THE INSIGHT. A backup has been created.",
        "next_step": (
            f"To delete this insight, call: apply_delete_insight("
            f"insight_id='{insight_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}')"
        ),
        "restore_info": (
            f"To restore after deletion, call: restore_insight_from_backup("
            f"backup_path='{backup_path}', customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_delete_insight(
    insight_id: str,
    confirmation_token: str,
    customer: str | None = None,
) -> str:
    """Delete an insight (WRITE OPERATION).

    You must first call preview_delete_insight to get the confirmation_token.
    A backup was already created during preview.

    Args:
        insight_id: The ID of the insight to delete.
        confirmation_token: Token from preview_delete_insight.
        customer: The customer name (tpp, dlg, danceone). Auto-detects from CWD if not provided.

    Returns:
        JSON with:
        - success: Whether the operation succeeded
        - deleted_insight_id: The ID of the deleted insight
        - backup_path: Path to the backup for potential restore
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current insight to verify token
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/visualizationObjects/{insight_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return json.dumps(
            {
                "success": False,
                "error": f"Insight '{insight_id}' not found.",
            },
            indent=2,
        )
    response.raise_for_status()
    data = response.json()

    title = data["data"]["attributes"].get("title", "")

    # Verify token
    token_data = f"delete_insight:{insight_id}:{title}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_insight",
            object_id=insight_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The insight may have changed since preview.",
                "message": "Please run preview_delete_insight again to get a new token.",
            },
            indent=2,
        )

    # Find the backup path for reference
    backup_dir = _get_backup_dir(customer_name)
    backup_files = sorted(
        backup_dir.glob(f"visualizationObject_{insight_id[:8]}_*.json"), reverse=True
    )
    backup_path = str(backup_files[0]) if backup_files else "unknown"

    # Delete via DELETE
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_insight",
            object_id=insight_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to delete insight: {e}",
            },
            indent=2,
        )

    # Log successful deletion
    _log_audit(
        customer=customer_name,
        operation="apply_delete_insight",
        object_id=insight_id,
        status="success",
        details={"title": title, "backup_path": backup_path},
    )

    return json.dumps(
        {
            "success": True,
            "deleted_insight_id": insight_id,
            "deleted_title": title,
            "backup_path": backup_path,
            "message": f"Successfully deleted insight '{title}'.",
            "restore_info": f"To restore, call: restore_insight_from_backup(backup_path='{backup_path}', customer='{customer_name}')",
        },
        indent=2,
    )


# =============================================================================
# WRITE OPERATIONS (Dashboards)
# =============================================================================


@mcp.tool()
def preview_create_dashboard(
    dashboard_id: str,
    title: str,
    insight_ids: list[str],
    customer: str | None = None,
    description: str | None = None,
    section_title: str | None = None,
    columns: int = 2,
) -> str:
    """Preview creating a new dashboard (READ-ONLY).

    Creates a simple dashboard with insights arranged in a grid.
    For complex layouts, use update_dashboard after creation.

    Args:
        dashboard_id: ID for the new dashboard (lowercase, underscores).
        title: Display title for the dashboard.
        insight_ids: List of insight IDs to include as widgets.
        customer: Customer name (auto-detects from CWD).
        description: Optional description.
        section_title: Optional title for the main section.
        columns: Number of columns (1-4). Insights fill left-to-right.

    Returns:
        JSON with dashboard_definition, confirmation_token, next_step.
    """

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)
    sdk = _get_sdk()

    # Validate columns
    if columns < 1 or columns > 4:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid columns value: {columns}. Must be 1-4.",
            },
            indent=2,
        )

    # Check if dashboard already exists
    existing = _get_dashboard_by_id(host, token, ws_id, dashboard_id)
    if existing is not None:
        return json.dumps(
            {
                "success": False,
                "error": f"Dashboard '{dashboard_id}' already exists. Use preview_update_dashboard instead.",
            },
            indent=2,
        )

    # Validate insights exist (can be empty for dashboard with no initial insights)
    if insight_ids:
        insights_valid, missing_insights = _validate_insights_exist(ws_id, insight_ids, sdk)
        if not insights_valid:
            return json.dumps(
                {
                    "success": False,
                    "error": "Some insights do not exist in the workspace.",
                    "missing_insights": missing_insights,
                },
                indent=2,
            )

    # Build dashboard content
    content = _build_dashboard_layout(insight_ids, columns, section_title)

    # Build the definition for token generation
    dashboard_definition = {
        "id": dashboard_id,
        "title": title,
        "description": description or "",
        "insight_ids": insight_ids,
        "columns": columns,
        "section_title": section_title,
    }

    # Generate confirmation token
    token_data = (
        f"create_dashboard:{dashboard_id}:{json.dumps(dashboard_definition, sort_keys=True)}"
    )
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview
    _log_audit(
        customer=customer_name,
        operation="preview_create_dashboard",
        object_id=dashboard_id,
        status="preview",
        details={"title": title, "insight_count": len(insight_ids)},
    )

    result = {
        "success": True,
        "action": "create_dashboard",
        "dashboard_definition": dashboard_definition,
        "content_preview": content,
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To create this dashboard, call: apply_create_dashboard("
            f"dashboard_id='{dashboard_id}', title='{title}', "
            f"insight_ids={insight_ids}, confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_create_dashboard(
    dashboard_id: str,
    title: str,
    insight_ids: list[str],
    confirmation_token: str,
    customer: str | None = None,
    description: str | None = None,
    section_title: str | None = None,
    columns: int = 2,
) -> str:
    """Create a new dashboard (WRITE OPERATION).

    You must first call preview_create_dashboard to get the confirmation_token.

    Args:
        dashboard_id: ID for the new dashboard (lowercase, underscores).
        title: Display title for the dashboard.
        insight_ids: List of insight IDs to include as widgets.
        confirmation_token: Token from preview_create_dashboard.
        customer: Customer name (auto-detects from CWD).
        description: Optional description.
        section_title: Optional title for the main section.
        columns: Number of columns (1-4).

    Returns:
        JSON with success status and dashboard_id.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Build and verify the definition matches token
    dashboard_definition = {
        "id": dashboard_id,
        "title": title,
        "description": description or "",
        "insight_ids": insight_ids,
        "columns": columns,
        "section_title": section_title,
    }

    token_data = (
        f"create_dashboard:{dashboard_id}:{json.dumps(dashboard_definition, sort_keys=True)}"
    )
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_create_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. Parameters may have changed since preview.",
                "message": "Please run preview_create_dashboard again to get a new token.",
            },
            indent=2,
        )

    # Build the dashboard content
    content = _build_dashboard_layout(insight_ids, columns, section_title)

    # Build the API payload
    payload = {
        "data": {
            "type": "analyticalDashboard",
            "id": dashboard_id,
            "attributes": {
                "title": title,
                "description": description or "",
                "content": content,
            },
        }
    }

    # Create the dashboard via POST
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/analyticalDashboards"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_create_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to create dashboard: {e}",
            },
            indent=2,
        )

    # Log successful creation
    _log_audit(
        customer=customer_name,
        operation="apply_create_dashboard",
        object_id=dashboard_id,
        status="success",
        details={"title": title, "insight_count": len(insight_ids)},
    )

    return json.dumps(
        {
            "success": True,
            "dashboard_id": dashboard_id,
            "title": title,
            "insight_count": len(insight_ids),
            "message": f"Successfully created dashboard '{title}'.",
        },
        indent=2,
    )


@mcp.tool()
def preview_update_dashboard(
    dashboard_id: str,
    customer: str | None = None,
    title: str | None = None,
    description: str | None = None,
    insight_ids: list[str] | None = None,
    add_insight_ids: list[str] | None = None,
    remove_insight_ids: list[str] | None = None,
) -> str:
    """Preview updating an existing dashboard (READ-ONLY).

    Creates a backup before showing changes.
    No changes are made. Use apply_update_dashboard to execute the update.

    Args:
        dashboard_id: The ID of the dashboard to update.
        customer: Customer name (auto-detects from CWD).
        title: New title (optional).
        description: New description (optional).
        insight_ids: Replace all insights with this list (optional).
        add_insight_ids: Add these insights to existing (optional).
        remove_insight_ids: Remove these insights from existing (optional).

    Returns:
        JSON with current state, changes, backup_path, and confirmation_token.
    """

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)
    sdk = _get_sdk()

    # Fetch current dashboard
    data = _get_dashboard_by_id(host, token, ws_id, dashboard_id)
    if data is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Dashboard '{dashboard_id}' not found. Use preview_create_dashboard to create a new one.",
            },
            indent=2,
        )

    # Create backup
    backup_path = _save_backup(customer_name, "analyticalDashboard", dashboard_id, data)

    current_attrs = data["data"]["attributes"]
    current_content = current_attrs.get("content", {})
    current_layout = current_content.get("layout", {})

    # Extract current insight IDs from layout
    current_insight_ids = []
    for section in current_layout.get("sections", []):
        for item in section.get("items", []):
            widget = item.get("widget", {})
            if widget.get("type") == "insight":
                insight_ref = widget.get("insight", {}).get("identifier", {})
                if insight_ref.get("id"):
                    current_insight_ids.append(insight_ref["id"])

    current_state = {
        "title": current_attrs.get("title", ""),
        "description": current_attrs.get("description", ""),
        "insight_ids": current_insight_ids,
    }

    # Build changes dict
    changes = {}
    if title is not None and title != current_state["title"]:
        changes["title"] = {"from": current_state["title"], "to": title}
    if description is not None and description != current_state["description"]:
        changes["description"] = {"from": current_state["description"], "to": description}

    # Calculate new insight list
    new_insight_ids = None
    if insight_ids is not None:
        # Replace all insights
        new_insight_ids = insight_ids
    elif add_insight_ids is not None or remove_insight_ids is not None:
        # Modify existing list
        new_insight_ids = list(current_insight_ids)
        if add_insight_ids:
            for iid in add_insight_ids:
                if iid not in new_insight_ids:
                    new_insight_ids.append(iid)
        if remove_insight_ids:
            new_insight_ids = [iid for iid in new_insight_ids if iid not in remove_insight_ids]

    if new_insight_ids is not None and new_insight_ids != current_insight_ids:
        # Validate new insights exist
        insights_to_validate = [iid for iid in new_insight_ids if iid not in current_insight_ids]
        if insights_to_validate:
            insights_valid, missing_insights = _validate_insights_exist(
                ws_id, insights_to_validate, sdk
            )
            if not insights_valid:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Some insights do not exist in the workspace.",
                        "missing_insights": missing_insights,
                    },
                    indent=2,
                )
        changes["insight_ids"] = {"from": current_insight_ids, "to": new_insight_ids}

    if not changes:
        return json.dumps(
            {
                "success": True,
                "message": "No changes specified.",
                "current": current_state,
            },
            indent=2,
        )

    # Generate confirmation token
    update_def = {
        "dashboard_id": dashboard_id,
        "changes": changes,
    }
    token_data = f"update_dashboard:{dashboard_id}:{json.dumps(update_def, sort_keys=True)}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview
    _log_audit(
        customer=customer_name,
        operation="preview_update_dashboard",
        object_id=dashboard_id,
        status="preview",
        details={"changes": list(changes.keys())},
    )

    result = {
        "success": True,
        "action": "update_dashboard",
        "dashboard_id": dashboard_id,
        "current": current_state,
        "changes": changes,
        "backup_path": str(backup_path),
        "confirmation_token": confirmation_token,
        "next_step": (
            f"To apply these changes, call: apply_update_dashboard("
            f"dashboard_id='{dashboard_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}', ...changed_params...)"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_update_dashboard(
    dashboard_id: str,
    confirmation_token: str,
    customer: str | None = None,
    title: str | None = None,
    description: str | None = None,
    insight_ids: list[str] | None = None,
    add_insight_ids: list[str] | None = None,
    remove_insight_ids: list[str] | None = None,
) -> str:
    """Update an existing dashboard (WRITE OPERATION).

    You must first call preview_update_dashboard to get the confirmation_token.
    A backup was already created during preview.

    Args:
        dashboard_id: The ID of the dashboard to update.
        confirmation_token: Token from preview_update_dashboard.
        customer: Customer name (auto-detects from CWD).
        title: New title (optional).
        description: New description (optional).
        insight_ids: Replace all insights with this list (optional).
        add_insight_ids: Add these insights to existing (optional).
        remove_insight_ids: Remove these insights from existing (optional).

    Returns:
        JSON with success status and changes applied.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current dashboard
    data = _get_dashboard_by_id(host, token, ws_id, dashboard_id)
    if data is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Dashboard '{dashboard_id}' not found.",
            },
            indent=2,
        )

    current_attrs = data["data"]["attributes"]
    current_content = current_attrs.get("content", {})
    current_layout = current_content.get("layout", {})

    # Extract current insight IDs
    current_insight_ids = []
    for section in current_layout.get("sections", []):
        for item in section.get("items", []):
            widget = item.get("widget", {})
            if widget.get("type") == "insight":
                insight_ref = widget.get("insight", {}).get("identifier", {})
                if insight_ref.get("id"):
                    current_insight_ids.append(insight_ref["id"])

    current_state = {
        "title": current_attrs.get("title", ""),
        "description": current_attrs.get("description", ""),
        "insight_ids": current_insight_ids,
    }

    # Build changes for token verification
    changes = {}
    if title is not None and title != current_state["title"]:
        changes["title"] = {"from": current_state["title"], "to": title}
    if description is not None and description != current_state["description"]:
        changes["description"] = {"from": current_state["description"], "to": description}

    # Calculate new insight list
    new_insight_ids = None
    if insight_ids is not None:
        new_insight_ids = insight_ids
    elif add_insight_ids is not None or remove_insight_ids is not None:
        new_insight_ids = list(current_insight_ids)
        if add_insight_ids:
            for iid in add_insight_ids:
                if iid not in new_insight_ids:
                    new_insight_ids.append(iid)
        if remove_insight_ids:
            new_insight_ids = [iid for iid in new_insight_ids if iid not in remove_insight_ids]

    if new_insight_ids is not None and new_insight_ids != current_insight_ids:
        changes["insight_ids"] = {"from": current_insight_ids, "to": new_insight_ids}

    # Verify token
    update_def = {
        "dashboard_id": dashboard_id,
        "changes": changes,
    }
    token_data = f"update_dashboard:{dashboard_id}:{json.dumps(update_def, sort_keys=True)}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_update_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The dashboard may have changed since preview.",
                "message": "Please run preview_update_dashboard again to get a new token.",
            },
            indent=2,
        )

    # Apply changes
    if title is not None:
        current_attrs["title"] = title
    if description is not None:
        current_attrs["description"] = description

    # Rebuild layout if insights changed
    if new_insight_ids is not None and new_insight_ids != current_insight_ids:
        # Preserve section title if it exists
        section_title = None
        if current_layout.get("sections"):
            section_title = current_layout["sections"][0].get("header", {}).get("title")
        # Rebuild content with new insights (default to 2 columns)
        current_attrs["content"] = _build_dashboard_layout(new_insight_ids, 2, section_title)

    # Update via PUT
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/analyticalDashboards/{dashboard_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_update_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to update dashboard: {e}",
                "message": "Backup was saved during preview. Use restore_dashboard_from_backup to restore if needed.",
            },
            indent=2,
        )

    # Log successful update
    _log_audit(
        customer=customer_name,
        operation="apply_update_dashboard",
        object_id=dashboard_id,
        status="success",
        details={"changes": list(changes.keys())},
    )

    return json.dumps(
        {
            "success": True,
            "dashboard_id": dashboard_id,
            "changes_applied": list(changes.keys()),
            "message": f"Successfully updated dashboard '{dashboard_id}'.",
        },
        indent=2,
    )


@mcp.tool()
def preview_delete_dashboard(
    dashboard_id: str,
    customer: str | None = None,
) -> str:
    """Preview deleting a dashboard (READ-ONLY).

    This creates a backup and shows what would be deleted.
    No changes are made. Use apply_delete_dashboard to execute the deletion.

    Args:
        dashboard_id: The ID of the dashboard to delete.
        customer: Customer name (auto-detects from CWD).

    Returns:
        JSON with dashboard_to_delete, confirmation_token, backup_path, warning.
    """
    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current dashboard
    data = _get_dashboard_by_id(host, token, ws_id, dashboard_id)
    if data is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Dashboard '{dashboard_id}' not found.",
            },
            indent=2,
        )

    # Create backup
    backup_path = _save_backup(customer_name, "analyticalDashboard", dashboard_id, data)

    current_attrs = data["data"]["attributes"]
    dashboard_title = current_attrs.get("title", "")

    # Generate confirmation token
    token_data = f"delete_dashboard:{dashboard_id}:{dashboard_title}"
    confirmation_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    # Log the preview
    _log_audit(
        customer=customer_name,
        operation="preview_delete_dashboard",
        object_id=dashboard_id,
        status="preview",
        details={"title": dashboard_title, "backup_path": str(backup_path)},
    )

    result = {
        "success": True,
        "action": "delete_dashboard",
        "dashboard_to_delete": {
            "id": dashboard_id,
            "title": dashboard_title,
        },
        "backup_path": str(backup_path),
        "confirmation_token": confirmation_token,
        "warning": "THIS WILL PERMANENTLY DELETE THE DASHBOARD. A backup has been created.",
        "next_step": (
            f"To delete this dashboard, call: apply_delete_dashboard("
            f"dashboard_id='{dashboard_id}', confirmation_token='{confirmation_token}', "
            f"customer='{customer_name}')"
        ),
        "restore_info": (
            f"To restore after deletion, call: restore_dashboard_from_backup("
            f"backup_path='{backup_path}', customer='{customer_name}')"
        ),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def apply_delete_dashboard(
    dashboard_id: str,
    confirmation_token: str,
    customer: str | None = None,
) -> str:
    """Delete a dashboard (WRITE OPERATION).

    You must first call preview_delete_dashboard to get the confirmation_token.
    A backup was already created during preview.

    Args:
        dashboard_id: The ID of the dashboard to delete.
        confirmation_token: Token from preview_delete_dashboard.
        customer: Customer name (auto-detects from CWD).

    Returns:
        JSON with success status, deleted_dashboard_id, backup_path.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Fetch current dashboard to verify token
    data = _get_dashboard_by_id(host, token, ws_id, dashboard_id)
    if data is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Dashboard '{dashboard_id}' not found.",
            },
            indent=2,
        )

    dashboard_title = data["data"]["attributes"].get("title", "")

    # Verify token
    token_data = f"delete_dashboard:{dashboard_id}:{dashboard_title}"
    expected_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]

    if confirmation_token != expected_token:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"reason": "token_mismatch"},
        )
        return json.dumps(
            {
                "success": False,
                "error": "Invalid confirmation token. The dashboard may have changed since preview.",
                "message": "Please run preview_delete_dashboard again to get a new token.",
            },
            indent=2,
        )

    # Find backup path
    backup_dir = _get_backup_dir(customer_name)
    backup_files = sorted(backup_dir.glob(f"analyticalDashboard_{dashboard_id[:8]}_*.json"))
    backup_path = str(backup_files[-1]) if backup_files else "unknown"

    # Delete via DELETE
    url = f"{host}/api/v1/entities/workspaces/{ws_id}/analyticalDashboards/{dashboard_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
    }

    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="apply_delete_dashboard",
            object_id=dashboard_id,
            status="error",
            details={"error": str(e)},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to delete dashboard: {e}",
            },
            indent=2,
        )

    # Log successful deletion
    _log_audit(
        customer=customer_name,
        operation="apply_delete_dashboard",
        object_id=dashboard_id,
        status="success",
        details={"title": dashboard_title, "backup_path": backup_path},
    )

    return json.dumps(
        {
            "success": True,
            "deleted_dashboard_id": dashboard_id,
            "deleted_title": dashboard_title,
            "backup_path": backup_path,
            "message": f"Successfully deleted dashboard '{dashboard_title}'.",
            "restore_info": (
                f"To restore, call: restore_dashboard_from_backup("
                f"backup_path='{backup_path}', customer='{customer_name}')"
            ),
        },
        indent=2,
    )


@mcp.tool()
def restore_dashboard_from_backup(
    backup_path: str,
    customer: str | None = None,
) -> str:
    """Restore a dashboard from a backup file (WRITE OPERATION).

    Use this to recover a deleted dashboard or undo changes.
    The backup_path is provided in the response of apply_* operations.

    Args:
        backup_path: Path to the backup file (from a previous write operation).
        customer: Customer name (auto-detects from CWD).

    Returns:
        JSON with success status and details.
    """
    import requests

    _load_env()
    host = os.getenv("GOODDATA_HOST")
    token = os.getenv("GOODDATA_TOKEN")

    if not host or not token:
        raise ValueError("GOODDATA_HOST and GOODDATA_TOKEN must be set")

    customer_name = _resolve_customer_name(customer)
    ws_id = _resolve_workspace_id(customer)

    # Read the backup file
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return json.dumps(
            {
                "success": False,
                "error": f"Backup file not found: {backup_path}",
            },
            indent=2,
        )

    with open(backup_file) as f:
        backup_data = json.load(f)

    # Verify backup type
    if backup_data.get("object_type") != "analyticalDashboard":
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid backup type: {backup_data.get('object_type')}. Expected 'analyticalDashboard'.",
            },
            indent=2,
        )

    original_data = backup_data.get("data", {})
    dashboard_id = backup_data.get("object_id")

    if not dashboard_id or not original_data:
        return json.dumps(
            {
                "success": False,
                "error": "Invalid backup file structure.",
            },
            indent=2,
        )

    # Check if dashboard exists (PUT for update) or needs to be created (POST)
    existing = _get_dashboard_by_id(host, token, ws_id, dashboard_id)

    url = f"{host}/api/v1/entities/workspaces/{ws_id}/analyticalDashboards"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.gooddata.api+json",
        "Content-Type": "application/vnd.gooddata.api+json",
    }

    try:
        if existing is not None:
            # Update existing
            url = f"{url}/{dashboard_id}"
            response = requests.put(url, headers=headers, json=original_data)
        else:
            # Create new
            response = requests.post(url, headers=headers, json=original_data)
        response.raise_for_status()
    except Exception as e:
        _log_audit(
            customer=customer_name,
            operation="restore_dashboard_from_backup",
            object_id=dashboard_id,
            status="error",
            details={"error": str(e), "backup_path": backup_path},
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to restore dashboard: {e}",
            },
            indent=2,
        )

    # Log successful restore
    _log_audit(
        customer=customer_name,
        operation="restore_dashboard_from_backup",
        object_id=dashboard_id,
        status="success",
        details={"backup_path": backup_path},
    )

    dashboard_title = original_data.get("data", {}).get("attributes", {}).get("title", "")

    return json.dumps(
        {
            "success": True,
            "dashboard_id": dashboard_id,
            "title": dashboard_title,
            "message": f"Successfully restored dashboard '{dashboard_title}' from backup.",
            "action": "updated" if existing else "created",
        },
        indent=2,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
