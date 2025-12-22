# GoodData MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

**Give Claude Code direct access to your GoodData analytics platform.**

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables AI assistants like Claude to interact with GoodData, allowing natural language queries of your business intelligence data. Also includes a standalone CLI for direct terminal access.

## Why Use This?

- **AI-Powered Analytics**: Ask Claude questions about your GoodData dashboards, metrics, and insights in natural language
- **Read-Only by Design**: All operations are strictly read-only, so there's no risk of modifying your data
- **Zero Configuration**: Works with Claude Code out of the box (just add your credentials)
- **Export Capabilities**: Export dashboards to PDF and visualizations to CSV/Excel
- **CLI Included**: Use the `gooddata` command directly from your terminal

## Features

| Feature | Description |
|---------|-------------|
| **List Resources** | Workspaces, dashboards, insights, metrics, datasets, users, groups |
| **Query Data** | Retrieve data from any insight/visualization |
| **Export Reports** | Export dashboards to PDF, visualizations to CSV/XLSX |
| **Explore Data Models** | Get the logical data model (LDM) for documentation |
| **User Management** | List users, groups, and group memberships |

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/aalexmrt/gooddata-mcp-server.git
cd gooddata-mcp-server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with MCP support
pip install -e ".[mcp]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your GoodData credentials:

```env
GOODDATA_HOST=https://your-org.cloud.gooddata.com
GOODDATA_TOKEN=your-api-token
GOODDATA_WORKSPACE=your-default-workspace-id
```

> **Getting an API Token**: Go to your GoodData profile settings and create a Personal Access Token.

### 3. Add to Claude Code

```bash
claude mcp add gooddata \
  -s user \
  -e GOODDATA_HOST="https://your-org.cloud.gooddata.com" \
  -e GOODDATA_TOKEN="your-api-token" \
  -e GOODDATA_WORKSPACE="your-workspace-id" \
  -- /path/to/gooddata-mcp-server/.venv/bin/python -m gooddata_cli.mcp_server
```

### 4. Start Using

In Claude Code, you can now ask questions like:
- "List all dashboards in my GoodData workspace"
- "What metrics are available?"
- "Show me the data from the Revenue Overview insight"
- "Export the Sales Dashboard to PDF"

---

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `list_workspaces` | List all available workspaces |
| `list_insights` | List all insights (visualizations) in a workspace |
| `list_dashboards` | List all dashboards in a workspace |
| `list_metrics` | List all metrics in a workspace |
| `list_datasets` | List all datasets in a workspace |
| `get_dashboard_insights` | Get all insights contained in a dashboard |
| `get_dashboard_filters` | Get all filters configured on a dashboard |
| `get_insight_metadata` | Get detailed metadata for an insight |
| `get_insight_data` | Get data from an insight |
| `get_logical_data_model` | Get the workspace's logical data model |
| `list_users` | List all users in the organization |
| `list_user_groups` | List all user groups |
| `get_user_group_members` | Get members of a specific group |
| `export_dashboard_pdf` | Export a dashboard to PDF |
| `export_visualization_csv` | Export a visualization to CSV |
| `export_visualization_xlsx` | Export a visualization to Excel |

### MCP Server Management

```bash
# Check server status
claude mcp list

# Remove the server
claude mcp remove gooddata
```

---

## CLI Usage

The package also includes a standalone CLI for direct terminal access.

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

# JSON output
gooddata list workspaces --json
```

### Query Data

```bash
# Get data from a specific insight
gooddata insight <insight_id> -w <workspace_id>
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

---

## Python API

You can also use the package programmatically:

```python
from gooddata_cli.query import list_workspaces, list_insights
from gooddata_cli.export import export_dashboard_pdf, export_visualization_tabular

# List workspaces
workspaces = list_workspaces()

# List insights
insights = list_insights(workspace_id="my_workspace")

# Export dashboard
path = export_dashboard_pdf("dashboard_id", workspace_id="my_workspace")
```

---

## Project Structure

```
gooddata-mcp-server/
├── .env.example           # Environment template
├── pyproject.toml         # Package configuration
├── README.md              # This file
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Contribution guidelines
└── src/gooddata_cli/
    ├── __init__.py        # Package exports
    ├── sdk.py             # SDK initialization
    ├── query.py           # Query operations
    ├── export.py          # Export operations
    ├── cli.py             # CLI entry point
    └── mcp_server.py      # MCP server for Claude Code
```

---

## Security Note

**All operations are read-only.** This MCP server cannot create, update, or delete any data in GoodData. Operations are limited to:

- Listing resources (workspaces, insights, dashboards, metrics, datasets, users, groups)
- Querying existing data
- Exporting reports to local files

---

## Dependencies

- [gooddata-sdk](https://pypi.org/project/gooddata-sdk/) - Official GoodData Python SDK
- [gooddata-pandas](https://pypi.org/project/gooddata-pandas/) - Pandas integration for GoodData
- [mcp](https://pypi.org/project/mcp/) - Model Context Protocol SDK
- [click](https://pypi.org/project/click/) - CLI framework
- [rich](https://pypi.org/project/rich/) - Terminal formatting
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable loading

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Related Projects

- [GoodData Python SDK](https://github.com/gooddata/gooddata-python-sdk) - Official GoodData SDK
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [Claude Code](https://claude.ai/code) - AI coding assistant
