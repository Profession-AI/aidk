#!/usr/bin/env python3
"""
Aidk Command Line Interface

This module provides a command-line interface for AIDK.

Available Commands:
    serve        Start the API server with configurable options to serve the app defined in main.py.
    run          Start the Chainlit UI interface for interactive chat.
    create       Create a new AIDK project with configuration files.
    info         Show detailed information about the AIDK installation.

Examples:
    
    # Start server with default settings
    aidk serve
    
    # Start server on custom host and port
    aidk serve --host 127.0.0.1 --port 8080
    
    # Start server with auto-reload enabled
    aidk serve --reload
    
    # Start server with multiple workers
    aidk serve --workers 4
    
    # Start Chainlit UI interface
    aidk run

"""
import typer
from typing import Optional
import sys
import os

# Add project root directory to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, _project_root)

_cli_app = typer.Typer(
    help="AIDK CLI - Command line interface for AIDK API server",
    no_args_is_help=True,
    add_completion=False
)

@_cli_app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0", 
        "--host", "-h", 
        help="Host address to bind the server to"
    ),
    port: int = typer.Option(
        8000, 
        "--port", "-p", 
        help="Port number to bind the server to"
    ),
    reload: bool = typer.Option(
        False, 
        "--reload", "-r", 
        help="Enable auto-reload when code changes are detected"
    ),
    workers: Optional[int] = typer.Option(
        None, 
        "--workers", "-w", 
        help="Number of worker processes to use (None for single process)"
    ),
    log_level: str = typer.Option(
        "info", 
        "--log-level", "-l", 
        help="Log level (debug, info, warning, error, critical)"
    ),
):
    """
    Start the AIDK API server.
    
    This command starts the API server with all configured endpoints
    for models, agents, authentication, and rate limiting.
    
    The server will be available at http://{host}:{port}
    
        """
    print(f"🚀 Starting AIDK server on {host}:{port}")
    print(f"📊 Log level: {log_level}")
    if reload:
        print("🔄 Auto-reload enabled")
    if workers:
        print(f"👥 Workers: {workers}")
    else:
        print("👤 Single process mode")
    
    try:
        from main import app as main_app
        main_app.serve(host=host, port=port, reload=reload, workers=workers, log_level=log_level)
    except ImportError as e:
        print(f"❌ Error importing main app: {e}")
        print("💡 Make sure main.py exists in the project root")
        raise typer.Exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise typer.Exit(1)


@_cli_app.command()
def create():
    """
    Create a new AIDK project with configuration files.
    
    This command creates the necessary files for a new AIDK project:
    - ai.yaml: Main configuration file
    - providers.keys: API keys for AI providers
    - observability.keys: API keys for observability platforms (optional)
    - main.py: Basic application file
    
    The command will prompt for configuration details interactively.
    """
    print("🚀 Creating new AIDK project...")
    print("=" * 50)
    
    # Check if files already exist
    existing_files = []
    if os.path.exists("ai.yaml"):
        existing_files.append("ai.yaml")
    if os.path.exists("providers.keys"):
        existing_files.append("providers.keys")
    if os.path.exists("observability.keys"):
        existing_files.append("observability.keys")
    if os.path.exists("main.py"):
        existing_files.append("main.py")
    
    if existing_files:
        print(f"⚠️  Warning: The following files already exist: {', '.join(existing_files)}")
        overwrite = input("Do you want to overwrite them? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("❌ Project creation cancelled.")
            return
    
    # Collect configuration
    config = {}
    
    print("\n📝 Configuration Setup")
    print("-" * 30)
    
    # Base model configuration
    print("\n🤖 Base Model Configuration")
    provider = input("AI Provider (openai, anthropic, google, etc.): ").strip()
    if provider:
        model_name = input(f"Model name (e.g., gpt-4o-mini for OpenAI): ").strip()
        api_key = input(f"API Key for {provider} [required]: ").strip()
        
        if api_key:
            config['base_model'] = {
                'provider': provider,
                'model': model_name or 'gpt-4o-mini'
            }
            # Write API key to providers.keys
            with open("providers.keys", "w") as f:
                f.write(f"{provider.upper()}={api_key}\n")
            print(f"✅ API key saved to providers.keys")
        else:
            print("⚠️  No API key provided, base_model will not be configured")
    
    # Prompts path
    prompts_path = input("\n📁 Prompts directory path, default: prompts]: ").strip()
    if prompts_path:
        config['prompts_path'] = prompts_path
    else:
        config['prompts_path'] = "prompts"
    
    # Supported files
    print("\n📄 Supported Files Configuration")
    text_files = input("Text file extensions (comma-separated, e.g., txt,py,md): ").strip()
    image_files = input("Image file extensions (comma-separated, e.g., png,jpg,jpeg): ").strip()
    
    if text_files or image_files:
        config['supported_files'] = {}
        if text_files:
            config['supported_files']['text'] = [ext.strip() for ext in text_files.split(',')]
        if image_files:
            config['supported_files']['image'] = [ext.strip() for ext in image_files.split(',')]
    
    # Default prompts
    print("\n💬 Default Prompts Configuration")
    
    # Get default prompts from Conf class
    from aidk.conf.conf import Conf
    default_prompts = Conf._DEFAULT_CONFIG['default_prompt']
    
    rag_prompt = input(f"RAG prompt template (default: {default_prompts['rag']}): ").strip()
    summary_prompt = input(f"Summary prompt template (default: {default_prompts['summary']}): ").strip()
    file_prompt = input(f"File upload prompt template (default: {default_prompts['file']}): ").strip()
    
    # Always include default_prompt section, using defaults if user didn't provide custom values
    config['default_prompt'] = {}
    config['default_prompt']['rag'] = rag_prompt if rag_prompt else default_prompts['rag']
    config['default_prompt']['summary'] = summary_prompt if summary_prompt else default_prompts['summary']
    config['default_prompt']['file'] = file_prompt if file_prompt else default_prompts['file']
    
    # Observability
    print("\n📊 Observability Configuration")
    obs_platforms = input("Observability platforms (comma-separated, e.g., logfire,langfuse): ").strip()
    
    if obs_platforms:
        platforms = [p.strip() for p in obs_platforms.split(',')]
        config['observability'] = platforms
        
        # Ask for API keys for observability platforms
        obs_keys = {}
        for platform in platforms:
            if platform.lower() == 'langfuse':
                # Langfuse requires both PUBLIC_KEY and SECRET_KEY
                public_key = input(f"LANGFUSE_PUBLIC_KEY for {platform}: ").strip()
                secret_key = input(f"LANGFUSE_SECRET_KEY for {platform}: ").strip()
                
                if public_key or secret_key:
                    obs_keys['LANGFUSE_PUBLIC_KEY'] = public_key
                    obs_keys['LANGFUSE_SECRET_KEY'] = secret_key
            else:
                # Other platforms use single API key
                key = input(f"API Key for {platform}: ").strip()
                if key:
                    obs_keys[platform.upper()] = key
        
        if obs_keys:
            with open("observability.keys", "w") as f:
                for key_name, key_value in obs_keys.items():
                    f.write(f"{key_name}={key_value}\n")
            print(f"✅ Observability keys saved to observability.keys")
    
    # Create ai.yaml
    import yaml
    with open("ai.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("✅ ai.yaml created")
    
    # Create providers.keys if it doesn't exist
    if not os.path.exists("providers.keys"):
        with open("providers.keys", "w") as f:
            f.write("# Add your API keys here\n")
            f.write("# Format: PROVIDER_NAME=your_api_key\n")
        print("✅ providers.keys created (empty)")
    
    # Create prompts directory and hello.prompt file
    prompts_dir = config.get('prompts_path', 'prompts')
    os.makedirs(prompts_dir, exist_ok=True)
    
    hello_prompt_content = '''<prompt response_type="text">
Hello AIDK! Please introduce yourself and tell me what you can do.
</prompt>'''
    
    hello_prompt_path = os.path.join(prompts_dir, "hello.prompt")
    with open(hello_prompt_path, "w") as f:
        f.write(hello_prompt_content)
    print(f"✅ {hello_prompt_path} created")
    
    main_py_content = '''from aidk.models import Model
from aidk.prompts import Prompt
# Initialize model
'''

    if provider == "":
        main_py_content += '''# Remember to set your provider key in the providers.keys file
provider = "" # set your provider (es. openai, anthropic, google, etc.)
model = "" # set your model (es. gpt-4o-mini, claude-3, gemini-1.5-flash, etc.)
model = Model(provider=provider, model=model)
'''
    else:
        main_py_content += f'''model = Model(provider="{provider}", model="{model_name or 'gpt-4o-mini'}")
'''
    
    main_py_content += '''# Load and use the hello prompt
prompt = Prompt(prompt_id="hello")
response = model.ask(prompt)
print(f"Response: {response['response']}")
'''

    with open("main.py", "w") as f:
        f.write(main_py_content)
    print("✅ main.py created")
    
    print("\n🎉 Project created successfully!")


@_cli_app.command()
def run():
    """
    Start the Chainlit UI interface.
    
    This command starts the Chainlit web interface for AIDK,
    providing an interactive chat interface for users.
    
    The UI will be available in your web browser after starting.
    """
    print("🚀 Starting AIDK Chainlit UI...")
    print("🌐 The UI will open in your web browser")
    print("📱 Use Ctrl+C to stop the server")
    
    try:
        import subprocess
        import sys
        
        # Run chainlit command
        cmd = [sys.executable, "-m", "chainlit", "run", "aidk/ui/ui.py", "-w"]
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Chainlit: {e}")
        print("💡 Make sure chainlit is installed: pip install chainlit")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Chainlit UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting Chainlit UI: {e}")
        raise typer.Exit(1)


@_cli_app.command()
def info():
    """
    Show detailed information about the AIDK installation.
    
    Displays comprehensive information about the AIDK installation,
    including available models, agents, and configuration.
    """
    print("📋 AIDK Information")
    print("=" * 50)
    
    # Version info
    try:
        import aidk
        print(f"Version: {getattr(aidk, '__version__', 'unknown')}")
    except ImportError:
        print("Version: unknown (not installed)")
    
    # Python info
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Project info
    print(f"Project root: {_project_root}")
    print(f"CLI file: {__file__}")
    
    # Check main.py
    main_py_path = os.path.join(_project_root, "main.py")
    if os.path.exists(main_py_path):
        print(f"✅ main.py found: {main_py_path}")
    else:
        print(f"❌ main.py not found: {main_py_path}")
    
    # Check for configuration files
    config_files = ["requirements.txt", "pyproject.toml", "setup.py"]
    print("\n📁 Configuration files:")
    for config_file in config_files:
        config_path = os.path.join(_project_root, config_file)
        if os.path.exists(config_path):
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}")
    
    print("\n🔗 Available commands:")
    print("  create    - Create a new AIDK project")
    print("  serve     - Start the API server")
    print("  run       - Start the Chainlit UI interface")
    print("  info      - Show detailed information")
    
# Export cli_app as alias for _cli_app for external imports
cli_app = _cli_app

if __name__ == "__main__":
    _cli_app()
