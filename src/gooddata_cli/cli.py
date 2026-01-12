"""CLI entry point for GoodData operations."""

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from gooddata_cli import query, export, sync

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """GoodData CLI - Query and export data from GoodData."""
    pass


# ============================================================================
# List Commands
# ============================================================================


@main.group()
def list():
    """List GoodData resources."""
    pass


@list.command("workspaces")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_workspaces(as_json: bool):
    """List all available workspaces."""
    try:
        workspaces = query.list_workspaces()

        if as_json:
            click.echo(json.dumps(workspaces, indent=2))
        else:
            table = Table(title="Workspaces")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")

            for ws in workspaces:
                table.add_row(ws["id"], ws["name"])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@list.command("insights")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_insights(workspace: str | None, as_json: bool):
    """List all insights in a workspace."""
    try:
        insights = query.list_insights(workspace)

        if as_json:
            click.echo(json.dumps(insights, indent=2))
        else:
            table = Table(title="Insights")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")

            for insight in insights:
                table.add_row(insight["id"], insight["title"])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@list.command("dashboards")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_dashboards(workspace: str | None, as_json: bool):
    """List all dashboards in a workspace."""
    try:
        dashboards = query.list_dashboards(workspace)

        if as_json:
            click.echo(json.dumps(dashboards, indent=2))
        else:
            table = Table(title="Dashboards")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")

            for db in dashboards:
                table.add_row(db["id"], db["title"])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@list.command("metrics")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_metrics(workspace: str | None, as_json: bool):
    """List all metrics in a workspace."""
    try:
        metrics = query.list_metrics(workspace)

        if as_json:
            click.echo(json.dumps(metrics, indent=2))
        else:
            table = Table(title="Metrics")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Format", style="yellow")

            for m in metrics:
                table.add_row(m["id"], m["title"], m.get("format") or "")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@list.command("datasets")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_datasets(workspace: str | None, as_json: bool):
    """List all datasets in a workspace."""
    try:
        datasets = query.list_datasets(workspace)

        if as_json:
            click.echo(json.dumps(datasets, indent=2))
        else:
            table = Table(title="Datasets")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")

            for ds in datasets:
                table.add_row(ds["id"], ds["title"])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# ============================================================================
# Query Commands
# ============================================================================


@main.command("insight")
@click.argument("insight_id")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def get_insight(insight_id: str, workspace: str | None, as_json: bool):
    """Get data from a specific insight."""
    try:
        data = query.get_insight_data(insight_id, workspace, as_dataframe=False)

        if as_json:
            click.echo(json.dumps(data, indent=2, default=str))
        else:
            console.print(data)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# ============================================================================
# Export Commands
# ============================================================================


@main.group()
def export_cmd():
    """Export dashboards and visualizations."""
    pass


# Rename to avoid conflict with export module
main.add_command(export_cmd, name="export")


@export_cmd.command("pdf")
@click.argument("dashboard_id")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("-o", "--output", help="Output file path")
def export_pdf(dashboard_id: str, workspace: str | None, output: str | None):
    """Export a dashboard to PDF."""
    try:
        path = export.export_dashboard_pdf(dashboard_id, workspace, output)
        console.print(f"[green]Exported to:[/green] {path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@export_cmd.command("csv")
@click.argument("visualization_id")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("-o", "--output", help="Output file path")
def export_csv(visualization_id: str, workspace: str | None, output: str | None):
    """Export a visualization to CSV."""
    try:
        path = export.export_visualization_tabular(
            visualization_id, workspace, output, format="CSV"
        )
        console.print(f"[green]Exported to:[/green] {path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@export_cmd.command("xlsx")
@click.argument("visualization_id")
@click.option("-w", "--workspace", help="Workspace ID")
@click.option("-o", "--output", help="Output file path")
def export_xlsx(visualization_id: str, workspace: str | None, output: str | None):
    """Export a visualization to Excel."""
    try:
        path = export.export_visualization_tabular(
            visualization_id, workspace, output, format="XLSX"
        )
        console.print(f"[green]Exported to:[/green] {path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# ============================================================================
# Sync Commands
# ============================================================================


@main.group()
def sync_cmd():
    """Sync GoodData artifacts to local cache."""
    pass


main.add_command(sync_cmd, name="sync")


@sync_cmd.command("all")
@click.option("--no-children", is_flag=True, help="Skip child workspaces")
@click.option("--only", multiple=True, help="Only sync specific artifacts (ldm, analytics_model, catalog)")
def sync_all(no_children: bool, only: tuple[str, ...]):
    """Sync all configured customers."""
    try:
        artifacts = list(only) if only else None
        sync.sync_all(
            include_children=not no_children,
            artifacts=artifacts,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@sync_cmd.command("customer")
@click.argument("customer_name")
@click.option("--no-children", is_flag=True, help="Skip child workspaces")
@click.option("--only", multiple=True, help="Only sync specific artifacts (ldm, analytics_model, catalog)")
def sync_customer(customer_name: str, no_children: bool, only: tuple[str, ...]):
    """Sync a specific customer (tpp, dlg, danceone)."""
    try:
        artifacts = list(only) if only else None
        sync.sync_customer(
            customer_name=customer_name,
            include_children=not no_children,
            artifacts=artifacts,
        )
        console.print(f"\n[green]Sync complete for {customer_name}![/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@sync_cmd.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def sync_status(as_json: bool):
    """Show sync status for all customers."""
    try:
        status = sync.get_sync_status()

        if as_json:
            click.echo(json.dumps(status, indent=2))
        else:
            for customer, info in status.items():
                console.print(f"\n[bold cyan]{customer.upper()}[/bold cyan]")
                console.print(f"  Project: {info['project_path']}")

                if info["parent"]:
                    parent = info["parent"]
                    if parent.get("synced_at"):
                        console.print(f"  [green]Parent:[/green] Last synced {parent['synced_at']}")
                    else:
                        console.print(f"  [yellow]Parent:[/yellow] {parent.get('status', 'unknown')}")

                if info["child"]:
                    child = info["child"]
                    if child.get("synced_at"):
                        console.print(f"  [green]Child:[/green] Last synced {child['synced_at']}")
                    else:
                        console.print(f"  [yellow]Child:[/yellow] {child.get('status', 'unknown')}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@sync_cmd.command("list")
def sync_list():
    """List configured customers."""
    try:
        customers = sync.get_customers()

        table = Table(title="Configured Customers")
        table.add_column("Customer", style="cyan")
        table.add_column("Workspace ID", style="green")
        table.add_column("Project Path", style="yellow")

        for name, info in customers.items():
            table.add_row(
                name,
                info["workspace_id"],
                info["project_path"],
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
