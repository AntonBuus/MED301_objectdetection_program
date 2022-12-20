Interactive Digital Card Exhibit
by Group 301 on Medialogy 3. semester | Hamza Majid Qureshi, Gabriel C.D. Walløe, Malthe Hansen, Anton Buus Hansen, Amanda Leithoff Johansen 

_____________________________________________________

# Table of Contents:
## 1. The Project

## 2. Prerequisties 

  ### 2.1 - The environment
  	2.1.1 Virtual Environment 
	
  ### 2.2 - requirement.txt
	
## 3. Run the Program 
  
_____________________________________________________

# 1. The Project: 
For the semester project, the group decided to create an exhibit that promotes social and game-based learning in collaboration with Thorvaldsens Mueseum. 

_____________________________________________________
# 2. How to use the program
The following section, will provide a guide on how to use the program

  ## 2.1 The environment
  
  To run the program, you must have a installation of Python 3.9.x or older. From our testing, Pytorch is not yet optimised for Python 3.10 and newer. If you have a new version of Python installed, we recommend creating a virtual environment
    
    	2.1.1 Virtual Environment 
		To create a virtual environment in Python, follow these steps:
    
    	On MacOS:
    
    	Step 1: 
    		Open the terminal and navigate to the directory where you want to create the virtual environment.
    
    	Step 2:
    		Type the following into the terminal:
    		**python3.9 -m venv env**

    	Step 3:
    		To activate the virtual Environment run the following command:
    		**source env/bin/activate**
    
    	On Windows:
    
    	Step 1:
    		Open the command prompt and navigate to the directory where you want to create the virtual environment.
    
    	Step 2:
    		Run the following command:
    		**python -m venv env**
    
    	Step 3:
    		Activate the virtual environment by running the following command:
   		**env\Scripts\activate.bat**
    
    	NOTE: 
    		**"env"** described above is the name of the virtual environment. This can be adjusted to your preference. 
    
  ## 2.2 Requirement.txt
  Once you have the right installation of Python, either be on the system or in a virtual environment, you need to install the required packages to run the program. 
  
  To do this:
 ##### Step 1: 
  Navigate to the Terminal on MacOS or Command Prompt on Windows
  
  ##### Step2: 
  If you have created a virtual environment, activate using above instructions
  
  ##### Step 3:
  Navigate to the directory where you have installed the files of this project
  
  ##### Step 4:
  Run the following command:
  pip install -r requirement.txt
  
	
__________________________________________

# 3. Run the Program

To run the program, run the **"main.py"** file. 

This can done through the terminal, by navigating to the the project directory and typing **"python3 main.py"**

or 

Running the file from your editor of choice. 

We recommend running it from the editor, as you have the option to decide which camera you want to use by adjusting **"video_index"**. 

    


