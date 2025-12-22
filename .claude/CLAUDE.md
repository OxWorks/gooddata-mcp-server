# GoodData SDK CLI

This project provides a CLI tool for interacting with GoodData via the official Python SDK.

## Important: Read-Only Operations

**All tools in this project are READ-ONLY.** This CLI does not create, update, or delete any data in GoodData. Operations are limited to:
- Listing resources (workspaces, insights, dashboards, metrics, datasets)
- Querying existing data
- Exporting reports to local files

Do not attempt to use this project for write operations (creating workspaces, modifying dashboards, updating metrics, etc.) as those capabilities are not implemented.

## Setup

1. Activate the virtual environment:
   ```bash
   cd /Users/alexmartinez/stackless_ws/gooddata-sdk
   source .venv/bin/activate
   ```

2. Ensure `.env` file exists with credentials:
   ```
   GOODDATA_HOST=https://your-org.cloud.gooddata.com
   GOODDATA_TOKEN=your-api-token
   GOODDATA_WORKSPACE=your-default-workspace-id
   ```

## CLI Commands

### List Resources

```bash
# List all workspaces
gooddata list workspaces

# List insights in a workspace
gooddata list insights -w <workspace_id>

# List dashboards
gooddata list dashboards -w <workspace_id>

# List metrics
gooddata list metrics -w <workspace_id>

# List datasets
gooddata list datasets -w <workspace_id>

# Add --json for JSON output
gooddata list workspaces --json
```

### Query Data

```bash
# Get data from a specific insight
gooddata insight <insight_id> -w <workspace_id>

# AI-powered natural language query
gooddata chat "What were total sales last month?" -w <workspace_id>

# Reset chat history before asking
gooddata chat "Start fresh question" -w <workspace_id> --reset
```

### Export

```bash
# Export dashboard to PDF
gooddata export pdf <dashboard_id> -w <workspace_id>

# Export visualization to CSV
gooddata export csv <visualization_id> -w <workspace_id>

# Export visualization to Excel
gooddata export xlsx <visualization_id> -w <workspace_id>

# Custom output path
gooddata export pdf <dashboard_id> -o ./my-report.pdf
```

## Python API (for scripts)

```python
from gooddata_cli import get_sdk, get_workspace_id
from gooddata_cli.query import list_workspaces, list_insights, ai_chat
from gooddata_cli.export import export_dashboard_pdf, export_visualization_tabular

# List workspaces
workspaces = list_workspaces()

# List insights
insights = list_insights(workspace_id="my_workspace")

# AI chat
response = ai_chat("What is our top selling product?", workspace_id="my_workspace")

# Export dashboard
path = export_dashboard_pdf("dashboard_id", workspace_id="my_workspace")
```

## MCP Server (for Claude Code)

This project includes an MCP server that exposes GoodData tools to Claude Code.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `mcp__gooddata__list_workspaces` | List all workspaces |
| `mcp__gooddata__list_insights` | List insights in a workspace |
| `mcp__gooddata__list_dashboards` | List dashboards in a workspace |
| `mcp__gooddata__get_dashboard_insights` | Get all insights from a specific dashboard |
| `mcp__gooddata__get_dashboard_filters` | Get all filters configured on a dashboard |
| `mcp__gooddata__list_metrics` | List metrics in a workspace |
| `mcp__gooddata__list_datasets` | List datasets in a workspace |
| `mcp__gooddata__get_logical_data_model` | Get/download the logical data model (LDM) |
| `mcp__gooddata__list_users` | List all users |
| `mcp__gooddata__list_user_groups` | List all user groups |
| `mcp__gooddata__get_user_group_members` | Get members of a group |
| `mcp__gooddata__get_insight_metadata` | Get detailed metadata (tags, creator, dates, etc.) |
| `mcp__gooddata__get_insight_data` | Get data from an insight |
| `mcp__gooddata__export_dashboard_pdf` | Export dashboard to PDF |
| `mcp__gooddata__export_visualization_csv` | Export visualization to CSV |
| `mcp__gooddata__export_visualization_xlsx` | Export visualization to Excel |

### MCP Server Management

```bash
# Check server status
claude mcp list

# Re-register the server
claude mcp add gooddata \
  -s user \
  -e GOODDATA_HOST="https://your-host" \
  -e GOODDATA_TOKEN="your-token" \
  -e GOODDATA_WORKSPACE="your-workspace" \
  -- /path/to/.venv/bin/python -m gooddata_cli.mcp_server

# Remove the server
claude mcp remove gooddata
```

## Project Structure

```
src/gooddata_cli/
├── __init__.py    # Package exports
├── sdk.py         # SDK initialization, .env loading
├── query.py       # Query operations (list, insight data, AI chat)
├── export.py      # Export operations (PDF, CSV, XLSX)
├── cli.py         # Click CLI entry point
└── mcp_server.py  # MCP server for Claude Code
```

## Dependencies

- `gooddata-sdk` - Official GoodData Python SDK
- `gooddata-pandas` - Pandas integration for DataFrames
- `python-dotenv` - Environment variable loading
- `click` - CLI framework
- `rich` - Terminal formatting
