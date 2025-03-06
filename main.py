import sys
import os

# Add the project directory to the Python path 
# Basically adding Skill-importance-project to path
# so it can search the whole space for our custom modules
project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

def main():
    print("Hello from skill-importance-project!")


if __name__ == "__main__":
    main()
