"""
REST API Server Entry Point

Starts the FastAPI server using Uvicorn.
This is the main entry point for running the REST API.

Usage:
    python run_api.py
    
    # Or with custom host/port:
    python run_api.py --host 0.0.0.0 --port 8082
    
    # Or via uvicorn directly:
    uvicorn src.api.app:app --reload
"""

import argparse
import sys
import logging

# Configure UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import uvicorn

from src.cross_cutting.logging import setup_logging
from config import get_default_config


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="Drug Image Analysis REST API Server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: 1 for GPU workloads)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = get_default_config()
    
    # Override with CLI args if provided
    host = args.host or config.api.host
    port = args.port or config.api.port
    workers = args.workers or config.api.workers
    
    logger.info(f"Starting Drug Image Analysis API...")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Reload: {args.reload}")
    
    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers if not args.reload else 1,  # reload doesn't support multiple workers
        reload=args.reload,
        log_level="debug" if args.debug else "info"
    )


if __name__ == "__main__":
    main()
