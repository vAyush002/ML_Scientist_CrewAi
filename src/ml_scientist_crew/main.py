#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from .crew import MlScientistCrew


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
     "path": r"C:\Users\anand\Downloads\titanic.csv"
    }

    try:
        MlScientistCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()


#python -m ml_scientist_crew.main
# run command
#for src cd C:\Users\anand\ml_scientist_crew\src


#for save run 1.->
#cd C:\Users\anand\ml_scientist_crew.venv\Scripts\Activate.ps1
#2 -> cd src python -m ml_scientist_crew.main


#final for a week 
#cd C:\Users\anand\ml_scientist_crew
# .venv\Scripts\Activate.ps1
#cd src python -m ml_scientist_crew.main



#1.powershell like ->PS C:\Users\anand\ml_scientist_crew>
#2. venv activate -> .venv\Scripts\Activate.ps1
#3.powersehll like after 2 is ->(ml-scientist-crew) PS C:\Users\anand\ml_scientist_crew>
#4. run project ->cd src
#python -m ml_scientist_crew.main

