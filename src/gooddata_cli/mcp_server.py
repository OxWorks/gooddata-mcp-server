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

                    current_metrics.append({
                        "local_identifier": local_id,
                        "metric_id": metric_id,
                        "title": metric_title,
                    })

                    if metric_id in seen_metric_ids:
                        duplicates.append({
                            "local_identifier": local_id,
                            "metric_id": metric_id,
                            "title": metric_title,
                            "duplicate_of": seen_metric_ids[metric_id],
                        })
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
                        duplicates.append({
                            "local_identifier": local_id,
                            "metric_id": metric_id,
                            "title": metric_title,
                            "duplicate_of": seen_metric_ids[metric_id],
                        })
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
        return json.dumps({
            "success": False,
            "error": "Invalid confirmation token. The insight may have changed since preview.",
            "message": "Please run preview_remove_duplicate_metrics again to get a new token.",
        }, indent=2)

    if not duplicates:
        return json.dumps({
            "success": False,
            "error": "No duplicate metrics found to remove.",
        }, indent=2)

    # Save backup BEFORE making any changes
    backup_path = _save_backup(customer_name, "visualizationObject", insight_id, data)

    # Remove duplicates from the measures bucket
    duplicate_local_ids = {d["local_identifier"] for d in duplicates}

    for bucket in buckets:
        if bucket.get("localIdentifier") == "measures":
            bucket["items"] = [
                item for item in bucket.get("items", [])
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
        return json.dumps({
            "success": False,
            "error": f"Failed to update insight: {e}",
            "backup_path": str(backup_path),
            "message": "Backup was saved. Use restore_insight_from_backup to restore if needed.",
        }, indent=2)

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

    return json.dumps({
        "success": True,
        "insight_id": insight_id,
        "backup_path": str(backup_path),
        "removed_duplicates": duplicates,
        "removed_count": len(duplicates),
        "new_metric_count": len(seen_metric_ids),
        "message": f"Successfully removed {len(duplicates)} duplicate metric(s). Backup saved.",
    }, indent=2)


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
        return json.dumps({
            "success": False,
            "error": f"Backup file not found: {backup_path}",
        }, indent=2)

    with open(backup_file) as f:
        backup = json.load(f)

    object_type = backup.get("object_type")
    object_id = backup.get("object_id")
    data = backup.get("data")
    backed_up_at = backup.get("backed_up_at")

    if object_type != "visualizationObject":
        return json.dumps({
            "success": False,
            "error": f"Unsupported object type for restore: {object_type}",
            "message": "Currently only visualizationObject restores are supported.",
        }, indent=2)

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
        return json.dumps({
            "success": False,
            "error": f"Failed to restore insight: {e}",
        }, indent=2)

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

    return json.dumps({
        "success": True,
        "object_id": object_id,
        "object_type": object_type,
        "restored_from": backup_path,
        "original_backup_time": backed_up_at,
        "message": f"Successfully restored {object_type} from backup.",
    }, indent=2)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
