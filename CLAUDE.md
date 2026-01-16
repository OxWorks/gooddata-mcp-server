# GoodData MCP Server

CLI and MCP server for interacting with GoodData via the official Python SDK.

## Operations

This project supports both **read** and **write** operations:

### Read Operations
- List resources (workspaces, insights, dashboards, metrics, datasets, users)
- Query insight data
- Export dashboards/visualizations (PDF, CSV, XLSX)
- Get logical data model (LDM)

### Write Operations
Write operations use a **two-phase commit pattern** for safety:
1. `preview_*` - Shows what would change (no modifications made)
2. `apply_*` - Executes the change after confirmation

All write operations automatically:
- Create backups before modification
- Log to audit trail (`~/.config/stackless/gooddata/<customer>/audit.jsonl`)
- Support rollback via `restore_*_from_backup`

## Setup

1. Activate the virtual environment:
   ```bash
   cd ~/stackless_ws/gooddata-mcp-server
   source .venv/bin/activate
   ```

2. Ensure `.env` file exists with credentials:
   ```
   GOODDATA_HOST=https://your-org.cloud.gooddata.com
   GOODDATA_TOKEN=your-api-token
   ```

3. Configure customers in `~/.config/gooddata/workspaces.yaml`

## CLI Commands

```bash
# List resources
gooddata list workspaces
gooddata list insights
gooddata list dashboards
gooddata list metrics
gooddata list datasets

# Query data
gooddata insight <insight_id>

# Export
gooddata export pdf <dashboard_id>
gooddata export csv <visualization_id>
gooddata export xlsx <visualization_id>

# Sync (cache GoodData artifacts locally)
gooddata sync all              # Sync all customers
gooddata sync customer <name>  # Sync specific customer
gooddata sync status           # Check sync status
gooddata sync list             # List configured customers

# Add --json for JSON output
gooddata list workspaces --json
```

## MCP Tools

### Read Operations

| Tool | Description |
|------|-------------|
| `list_workspaces` | List all workspaces |
| `list_insights` | List insights in a workspace |
| `list_dashboards` | List dashboards in a workspace |
| `get_dashboard_insights` | Get insights from a dashboard |
| `get_dashboard_filters` | Get filters on a dashboard |
| `list_metrics` | List metrics in a workspace |
| `list_datasets` | List datasets in a workspace |
| `get_logical_data_model` | Get the LDM |
| `list_users` | List all users |
| `list_user_groups` | List user groups |
| `get_user_group_members` | Get members of a group |
| `get_insight_metadata` | Get insight metadata (tags, dates, etc.) |
| `get_insight_data` | Get data from an insight |
| `get_metric` | Get metric definition (MAQL, format) |
| `list_visualization_types` | List supported viz types |

### Export Operations

| Tool | Description |
|------|-------------|
| `export_dashboard_pdf` | Export dashboard to PDF |
| `export_visualization_csv` | Export visualization to CSV |
| `export_visualization_xlsx` | Export visualization to Excel |

### Write Operations (Metrics)

| Tool | Description |
|------|-------------|
| `preview_create_metric` | Preview creating a metric |
| `apply_create_metric` | Create a metric |
| `preview_update_metric` | Preview updating a metric |
| `apply_update_metric` | Update a metric |
| `preview_delete_metric` | Preview deleting a metric |
| `apply_delete_metric` | Delete a metric |
| `restore_metric_from_backup` | Restore from backup |

### Write Operations (Insights)

| Tool | Description |
|------|-------------|
| `preview_create_insight` | Preview creating an insight |
| `apply_create_insight` | Create an insight |
| `preview_update_insight` | Preview updating an insight |
| `apply_update_insight` | Update an insight |
| `preview_delete_insight` | Preview deleting an insight |
| `apply_delete_insight` | Delete an insight |
| `preview_remove_duplicate_metrics` | Preview removing duplicates |
| `apply_remove_duplicate_metrics` | Remove duplicate metrics |
| `restore_insight_from_backup` | Restore from backup |

## Project Structure

```
src/gooddata_cli/
├── __init__.py      # Package exports
├── sdk.py           # SDK initialization, .env loading
├── query.py         # Query operations (list, insight data)
├── export.py        # Export operations (PDF, CSV, XLSX)
├── sync.py          # Sync operations (cache artifacts locally)
├── cli.py           # Click CLI entry point
└── mcp_server.py    # MCP server with all tools
```

## Development

```bash
# Install dependencies
uv pip install -e ".[dev,mcp]"

# Lint and format
ruff check --fix .
ruff format .

# Run tests
pytest -xvs
```

## Dependencies

- `gooddata-sdk` - Official GoodData Python SDK
- `gooddata-pandas` - Pandas integration
- `python-dotenv` - Environment variable loading
- `click` - CLI framework
- `rich` - Terminal formatting
- `mcp` - MCP server framework
