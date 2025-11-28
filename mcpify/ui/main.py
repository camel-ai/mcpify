"""
Main entry point for the MCPify Streamlit UI application.

This module provides the main Streamlit application for repository analysis
and MCP server configuration generation.
"""

import os
import subprocess
import sys


def start_ui() -> None:
    """
    Start the Streamlit UI application.

    This function imports and runs the main UI application.
    """
    try:
        # Get the path to the streamlit app
        import mcpify.ui.app

        app_path = os.path.abspath(mcpify.ui.app.__file__)

        print("🚀 Starting MCPify Repository Analyzer UI...")
        print("🌐 Navigate to: http://localhost:8501")
        print("Press Ctrl+C to stop")

        # Run streamlit as a subprocess
        # We use sys.executable -m streamlit to ensure we use the same python environment
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]

        # Pass through any additional arguments
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])

        subprocess.run(cmd, check=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install UI dependencies with: pip install 'mcpify[ui]'")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 MCPify UI stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Streamlit exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    start_ui()
