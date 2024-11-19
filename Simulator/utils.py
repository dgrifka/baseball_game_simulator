import os
import subprocess
import sys
import importlib

def reload_module_from_file(file_name, branch = 'main'):
    """
    Updates and reloads the module from the specified file path in Google Colab after pulling the latest changes from GitHub.
    This is useful for ensuring the Colab environment uses the most recent code version without restarting the runtime.

    Parameters:
    - file_path (str): Path to the Python file containing the module code.

    The function changes the current working directory to '/content/baseball_game_simulator', pulls the latest updates from the 'main'
    branch of the Git repository, reads the contents of the specified file, and executes the code in a separate namespace.
    It then updates the current namespace with the new definitions.
    """
    # Ensure prerequisites
    assert isinstance(file_name, str), "file_path must be a string"

    # Change directory and pull updates from Git
    os.chdir('/content/baseball_game_simulator')
    result = subprocess.run(['git', 'pull', 'origin', branch], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the output and error (if any) from the Git pull command
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Read the contents of the file
    with open(f"/content/baseball_game_simulator/{file_name}.py", 'r') as file:
        code = file.read()

    # Execute the code in a separate namespace
    namespace = {}
    exec(code, namespace)

    # Update the current namespace with the new definitions
    globals().update(namespace)
      

def reload_functions_from_module(module_name, function_names, branch):
    """
    Updates and reloads specified functions from a given module in Google Colab after pulling the latest changes from GitHub. 
    This is useful for ensuring the Colab environment uses the most recent code version without restarting the runtime.

    Parameters:
    - module_name (str): Name of the module to reload.
    - function_names (list): List of function names as strings to import into the global namespace after reloading the module.

    The function changes the current working directory to '/content/baseball_game_simulator', pulls the latest updates from the 'main' 
    branch of the Git repository, reloads the module, and then imports the specified functions into the global namespace, making them 
    directly callable.
    """
    # Ensure prerequisites
    assert isinstance(module_name, str), "module_name must be a string"
    assert isinstance(function_names, list) and all(isinstance(name, str) for name in function_names), "function_names must be a list of strings"

    # Change directory and pull updates from Git
    os.chdir('/content/baseball_game_simulator')
    result = subprocess.run(['git', 'pull', 'origin', branch], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print the output and error (if any) from the Git pull command
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Import and reload the module
    module = importlib.import_module(module_name)
    importlib.reload(module)
    
    # Import specified functions from the module into the global namespace
    for name in function_names:
        # Set the function in the global namespace
        globals()[name] = getattr(module, name)
