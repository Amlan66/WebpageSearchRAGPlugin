"""
RAG Chrome Plugin - Main Entry Point
Run this file to start the agent server
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentCP import app, log
import uvicorn


def main():
    """Main entry point for the RAG Chrome Plugin agent"""
    print("="*60)
    print("🚀 RAG Chrome Plugin - Starting Agent...")
    print("="*60)
    
    log("main", "Initializing agent server...")
    
    # Check for required files
    required_files = [
        'perceptionCP.py',
        'decisionCP.py',
        'actionCP.py',
        'memoryCP.py',
        'toolsCP.py',
        'modelsCP.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("   Create .env with: GEMINI_API_KEY=your_key")
    
    print("\n📋 Server Configuration:")
    print("   - Host: 127.0.0.1")
    print("   - Port: 8000")
    print("   - Docs: http://127.0.0.1:8000/docs")
    print("\n💡 Next steps:")
    print("   1. Load Chrome extension from chrome_extension/ folder")
    print("   2. Open extension popup to see stats")
    print("   3. Visit a webpage and click 'Index Current Page'")
    print("   4. Try searching!")
    print("\n" + "="*60 + "\n")
    
    # Start the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
