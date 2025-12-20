"""Query operations for GoodData."""

from typing import Any

from gooddata_cli.sdk import get_sdk, get_workspace_id


def list_workspaces() -> list[dict[str, str]]:
    """List all available workspaces.

    Returns:
        List of workspace dictionaries with id and name.
    """
    sdk = get_sdk()
    workspaces = sdk.catalog_workspace.list_workspaces()

    return [{"id": ws.id, "name": ws.name} for ws in workspaces]


def list_insights(workspace_id: str | None = None) -> list[dict[str, str]]:
    """List all insights (visualizations) in a workspace.

    Args:
        workspace_id: Optional workspace ID override.

    Returns:
        List of insight dictionaries with id and title.
    """
    sdk = get_sdk()
    ws_id = get_workspace_id(workspace_id)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    return [{"id": viz.id, "title": viz.title} for viz in am.analytics.visualization_objects]


def list_dashboards(workspace_id: str | None = None) -> list[dict[str, str]]:
    """List all dashboards in a workspace.

    Args:
        workspace_id: Optional workspace ID override.

    Returns:
        List of dashboard dictionaries with id and title.
    """
    sdk = get_sdk()
    ws_id = get_workspace_id(workspace_id)

    am = sdk.catalog_workspace_content.get_declarative_analytics_model(ws_id)

    return [{"id": db.id, "title": db.title} for db in am.analytics.analytical_dashboards]


def list_metrics(workspace_id: str | None = None) -> list[dict[str, Any]]:
    """List all metrics in a workspace.

    Args:
        workspace_id: Optional workspace ID override.

    Returns:
        List of metric dictionaries with id, title, and format.
    """
    sdk = get_sdk()
    ws_id = get_workspace_id(workspace_id)

    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)

    return [
        {"id": m.id, "title": m.title, "format": getattr(m, "format", None)}
        for m in catalog.metrics
    ]


def list_datasets(workspace_id: str | None = None) -> list[dict[str, str]]:
    """List all datasets in a workspace.

    Args:
        workspace_id: Optional workspace ID override.

    Returns:
        List of dataset dictionaries with id and title.
    """
    sdk = get_sdk()
    ws_id = get_workspace_id(workspace_id)

    catalog = sdk.catalog_workspace_content.get_full_catalog(ws_id)

    return [{"id": ds.id, "title": ds.title} for ds in catalog.datasets]


def get_insight_data(
    insight_id: str,
    workspace_id: str | None = None,
    as_dataframe: bool = False,
) -> Any:
    """Get data from a specific insight.

    Args:
        insight_id: The insight ID to query.
        workspace_id: Optional workspace ID override.
        as_dataframe: If True, return as pandas DataFrame.

    Returns:
        Insight data as dict or DataFrame.
    """
    sdk = get_sdk()
    ws_id = get_workspace_id(workspace_id)

    if as_dataframe:
        from gooddata_pandas import GoodPandas

        gp = GoodPandas(sdk.client.endpoint, sdk.client.token)
        return gp.data_frames(ws_id).for_insight(insight_id)
    else:
        insight = sdk.insights.get_insight(ws_id, insight_id)
        result = sdk.compute.for_insight(ws_id, insight)
        return {
            "headers": [h.header_value for h in result.headers],
            "data": result.data,
        }
