"""
Camel-AI based project detector.

This module contains the CamelDetector class that uses camel-ai's ChatAgent
framework for intelligent project analysis and tool detection.
"""

import json
import os
from pathlib import Path
from typing import Any

from .base import BaseDetector
from .types import DetectionResult, ProjectInfo, ToolSpec

try:
    from camel.agents import ChatAgent
    from camel.configs import ChatGPTConfig
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType

    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False


class CamelDetector(BaseDetector):
    """Camel-AI based project detector using ChatAgent framework."""

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs: Any) -> None:
        """Initialize the detector with Camel-AI ChatAgent."""
        # super().__init__(**kwargs)

        if not CAMEL_AVAILABLE:
            raise ImportError(
                "camel-ai is required for CamelDetector. "
                "Install it with: pip install camel-ai"
            )

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key is required for CamelDetector. "
                "Set OPENAI_API_KEY environment variable."
            )

        self.model_name = model_name
        self.agent: ChatAgent | None = None
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the ChatAgent with correct API usage."""
        # Create model using ModelFactory
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict=ChatGPTConfig(temperature=0.1).as_dict(),
        )

        # Define the system prompt
        system_message = """You are an expert software engineer specializing in API analysis and tool detection.
Your task is to analyze projects and identify all tools/APIs/commands that can be exposed
as MCP (Model Context Protocol) tools.

You excel at:
- Understanding code structure and patterns
- Identifying CLI commands and their arguments
- Recognizing web API endpoints
- Finding reusable functions and methods
- Extracting parameter information and types

Always provide detailed, accurate analysis in JSON format."""

        # Create ChatAgent using the new API
        self.agent = ChatAgent(
            system_message=system_message, model=model, message_window_size=10
        )

    def _detect_tools(
        self, project_path: Path, project_info: ProjectInfo
    ) -> list[ToolSpec]:
        """Use Camel-AI ChatAgent to detect tools/APIs in the project."""
        if not self.agent:
            raise ValueError("Camel-AI agent not initialized")

        # Gather project context
        context = self._gather_project_context(project_path, project_info)

        # Use ChatAgent to analyze and detect tools
        tools_data = self._agent_detect_tools(context, project_info)

        # Convert to ToolSpec objects
        tools = []
        for tool_data in tools_data:
            tools.append(
                ToolSpec(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    args=tool_data.get("args", []),
                    parameters=tool_data.get("parameters", []),
                )
            )

        return tools

    def _gather_project_context(
        self, project_path: Path, project_info: ProjectInfo
    ) -> str:
        """Gather comprehensive project context for agent analysis."""
        context_parts = []

        # Basic project info
        context_parts.append(f"Project Name: {project_info.name}")
        context_parts.append(f"Project Type: {project_info.project_type}")
        context_parts.append(f"Description: {project_info.description}")

        # Dependencies
        if project_info.dependencies:
            deps_str = ", ".join(project_info.dependencies)
            context_parts.append(f"Dependencies: {deps_str}")

        # README content (truncated)
        if project_info.readme_content:
            readme_excerpt = project_info.readme_content[:1500]
            context_parts.append(f"README:\n{readme_excerpt}")

        # Directory structure
        structure = self._get_directory_structure(project_path)
        context_parts.append(f"Directory Structure:\n{structure}")

        # Code samples from main files
        context_parts.append("\nKey Code Files:")
        for main_file in project_info.main_files[:5]:  # Limit to first 5 files
            file_path = project_path / main_file
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()[:2500]  # First 2500 chars
                    context_parts.append(f"\n=== {main_file} ===\n{content}")
            except Exception as e:
                context_parts.append(f"\n=== {main_file} ===\nError: {e}")

        return "\n".join(context_parts)

    def _get_directory_structure(self, project_path: Path, max_depth: int = 2) -> str:
        """Get a simplified directory structure."""
        structure_lines = []

        def walk_dir(path: Path, depth: int = 0, prefix: str = "") -> None:
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                for i, item in enumerate(items):
                    if item.name.startswith("."):
                        continue

                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    structure_lines.append(f"{prefix}{current_prefix}{item.name}")

                    if item.is_dir() and depth < max_depth:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        walk_dir(item, depth + 1, next_prefix)
            except PermissionError:
                pass

        walk_dir(project_path)
        return "\n".join(structure_lines[:20])  # Limit output

    def _agent_detect_tools(
        self, context: str, project_info: ProjectInfo
    ) -> list[dict[str, Any]]:
        """Use ChatAgent to analyze project and detect tools."""

        user_prompt = f"""Analyze this project and identify all possible tools/APIs that can be exposed as MCP tools:

{context}

Based on the project type ({project_info.project_type}) and the code analysis, identify all tools that could be exposed.

Consider these aspects:
1. CLI Commands: Look for argparse, click, typer patterns and their arguments
2. Web APIs: Identify Flask/Django/FastAPI routes and endpoints
3. Functions: Find public functions that could be called as tools
4. Scripts: Identify executable scripts and their parameters
5. Interactive Commands: Look for input/menu-driven functionality

For each tool, provide detailed information:
- name: Clear, descriptive name (snake_case)
- description: What the tool does and its purpose
- args: Command structure or API call pattern
- parameters: Detailed parameter specs with types and descriptions

Return your analysis as a JSON array with this EXACT format:
[
  {{
    "name": "tool_name",
    "description": "Clear description of what this tool does",
    "args": ["command", "--flag", "{{parameter}}"],
    "parameters": [
      {{
        "name": "parameter",
        "type": "string|integer|number|boolean|array",
        "description": "Parameter description",
        "required": true
      }}
    ]
  }}
]

Focus on practical, usable tools that provide real value."""

        try:
            # Send message to agent using the correct API
            user_message = BaseMessage.make_user_message(
                role_name="User", content=user_prompt
            )

            if self.agent is None:
                raise ValueError("Camel-AI agent not initialized")
            response = self.agent.step(user_message)
            content = response.msg.content.strip()

            # Extract JSON from response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx != 0:
                json_content = content[start_idx:end_idx]
            else:
                json_content = content

            tools_data = json.loads(json_content)

            # Validate the structure
            if not isinstance(tools_data, list):
                print("Warning: Agent returned non-list response, wrapping in list")
                tools_data = [tools_data] if tools_data else []

            # Validate each tool has required fields
            validated_tools = []
            for tool in tools_data:
                if isinstance(tool, dict) and "name" in tool and "description" in tool:
                    # Set defaults for missing fields
                    if "args" not in tool:
                        tool["args"] = []
                    if "parameters" not in tool:
                        tool["parameters"] = []
                    validated_tools.append(tool)
                else:
                    print(f"Warning: Skipping invalid tool specification: {tool}")

            return validated_tools

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse agent response as JSON: {e}")
            print(f"Raw response: {content}")
            return []
        except Exception as e:
            print(f"Error: Camel-AI tool detection failed: {e}")
            return []

    def _enhance_tool_with_agent(self, tool: ToolSpec, context: str) -> ToolSpec:
        """Use ChatAgent to enhance a single tool's specification."""
        try:
            prompt = f"""
            Enhance this tool specification with better descriptions and details:

            Tool: {tool.name}
            Current Description: {tool.description}
            Args: {tool.args}
            Parameters: {tool.parameters}

            Project Context (excerpt):
            {context[:800]}

            Provide an enhanced version with:
            1. A clear, professional description
            2. Better parameter descriptions with proper types
            3. Usage examples if helpful

            Return as JSON in this exact format:
            {{
              "name": "{tool.name}",
              "description": "Enhanced description",
              "args": {tool.args},
              "parameters": [
                {{
                  "name": "param_name",
                  "type": "string|integer|number|boolean|array",
                  "description": "Clear parameter description",
                  "required": true
                }}
              ]
            }}
            """

            user_message = BaseMessage.make_user_message(
                role_name="User", content=prompt
            )
            if self.agent is None:
                raise ValueError("Camel-AI agent not initialized")
            response = self.agent.step(user_message)
            enhanced_data = json.loads(response.msg.content)

            return ToolSpec(
                name=enhanced_data["name"],
                description=enhanced_data["description"],
                args=enhanced_data.get("args", tool.args),
                parameters=enhanced_data.get("parameters", tool.parameters),
            )

        except Exception as e:
            print(f"Warning: Failed to enhance tool {tool.name}: {e}")
            return tool

    def _detect_from_content(self, code_content: str) -> "DetectionResult":
        """
        Analyze code content and return detection result.

        Args:
            code_content: The code content to analyze (from GitIngest)

        Returns:
            DetectionResult with detected tools and project info
        """
        # Extract basic project info from content
        project_info = self._extract_project_info_from_content(code_content)

        # Use Camel-AI agent to detect tools
        tools_data = self._agent_detect_tools(code_content, project_info)

        # Convert to ToolSpec objects
        tools = []
        for tool_data in tools_data:
            try:
                tool = ToolSpec(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    args=tool_data.get("args", []),
                    parameters=tool_data.get("parameters", []),
                )
                tools.append(tool)
            except Exception as e:
                print(
                    f"Warning: Failed to create tool spec for {tool_data.get('name', 'unknown')}: {e}"
                )

        # Generate appropriate backend config
        backend_config = self._generate_backend_config_from_content(project_info)

        return DetectionResult(
            project_info=project_info,
            tools=tools,
            backend_config=backend_config,
            confidence_score=0.9,  # Higher confidence for Camel-AI
        )

    def _extract_project_info_from_content(self, code_content: str) -> ProjectInfo:
        """Extract project information from code content."""
        # Same logic as OpenaiDetector
        project_type = "library"
        if any(
            pattern in code_content
            for pattern in [
                "argparse",
                "ArgumentParser",
                "add_argument",
                "click",
                "typer",
            ]
        ):
            project_type = "commandline"
        elif any(
            pattern in code_content
            for pattern in ["fastapi", "flask", "django", "app.run", "@app.route"]
        ):
            project_type = "web"

        dependencies = []
        import_lines = [
            line
            for line in code_content.split("\n")
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        for line in import_lines:
            if "import" in line:
                parts = line.split()
                if len(parts) >= 2:
                    if parts[0] == "import":
                        dependencies.append(parts[1].split(".")[0])
                    elif parts[0] == "from":
                        dependencies.append(parts[1].split(".")[0])

        standard_libs = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "re",
            "pathlib",
            "typing",
            "abc",
        }
        dependencies = [dep for dep in set(dependencies) if dep not in standard_libs]

        return ProjectInfo(
            name="detected-project",
            description="Project analyzed from code content using Camel-AI",
            main_files=["main.py"],
            readme_content="",
            project_type=project_type,
            dependencies=dependencies,
        )

    def _generate_backend_config_from_content(
        self, project_info: ProjectInfo
    ) -> dict[str, Any]:
        """Generate backend configuration based on project info."""
        if project_info.project_type == "commandline":
            return {
                "type": "commandline",
                "config": {"command": "python3", "args": ["main.py"], "cwd": "."},
            }
        elif project_info.project_type == "web":
            return {
                "type": "http",
                "config": {"base_url": "http://localhost:8000", "timeout": 30},
            }
        else:
            return {
                "type": "commandline",
                "config": {
                    "command": "python3",
                    "args": ["main.py"],
                    "cwd": ".",
                },
            }
