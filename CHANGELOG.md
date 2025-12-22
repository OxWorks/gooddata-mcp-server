# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `get_dashboard_filters` MCP tool - Retrieve all attribute and date filters configured on a dashboard
- LICENSE file (MIT)
- CONTRIBUTING.md with development guidelines
- GitHub issue and PR templates
- GitHub Actions CI workflow

### Changed
- Improved README with badges, better structure, and SEO keywords

## [0.1.0] - 2024-12-18

### Added
- Initial release of GoodData MCP Server
- MCP tools for Claude Code integration:
  - `list_workspaces` - List all workspaces
  - `list_insights` - List insights in a workspace
  - `list_dashboards` - List dashboards in a workspace
  - `list_metrics` - List metrics in a workspace
  - `list_datasets` - List datasets in a workspace
  - `get_dashboard_insights` - Get insights from a dashboard
  - `get_insight_metadata` - Get detailed insight metadata
  - `get_insight_data` - Query data from an insight
  - `get_logical_data_model` - Get workspace LDM
  - `list_users` - List organization users
  - `list_user_groups` - List user groups
  - `get_user_group_members` - Get group members
  - `export_dashboard_pdf` - Export dashboard to PDF
  - `export_visualization_csv` - Export to CSV
  - `export_visualization_xlsx` - Export to Excel
- CLI interface with `gooddata` command
- Python API for programmatic access
- Read-only design for security

[Unreleased]: https://github.com/aalexmrt/gooddata-mcp-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aalexmrt/gooddata-mcp-server/releases/tag/v0.1.0
