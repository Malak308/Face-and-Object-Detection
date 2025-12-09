"""
Main application entry point
Run this file to start the Computer Vision Detection System GUI
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gui.main_window import ComputerVisionGUI


def main():
    """Main application entry point"""
    app = ComputerVisionGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
