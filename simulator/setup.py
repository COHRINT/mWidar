import os
import platform
import subprocess

#activate and run simulateTracks in venv

def activate_venv():
	#run in existing virtual environment
	if os.path.exists("venv"):
		print("Running venv")
		#determine the path to the Python interpreter within the virtual environment
		if platform.system() == "Windows":
			python_executable = os.path.join("venv", "Scripts", "python.exe")
		else:
			python_executable = os.path.join("venv", "bin", "python")
		#run simulateTracks.py using the virtual environment's Python interpreter
		print("running simulator")
		subprocess.run([python_executable, "simulateTracks.py",
			"-o", "[-4,50,2,1,0,0]", "[130,120,-1,-2,-1,0]",
			"-s", "true",
			"-T", "true"])
	else:
		print("No virtual environment found. Run the setup script first.")

def deactivate_venv():
		if platform.system() == "Windows":
			os.system("venv\\Scripts\\deactivate")
		else:
			os.system("deactivate")

if __name__== "__main__":
	activate_venv()
	deactivate_venv()
