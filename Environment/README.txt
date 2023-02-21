Notes on how to create the virtual environment

#Load a python version that you want to use
>module load python/3.10.7
#change to directory where you want to story directory files
>mkdir Environment
>cd Environment
#create environment
>virtualenv env_WDTheory
#activate the environment
>source Environment/env_WDTheory/bin/activate
#check where pip points to 
>which pip
#Output: Environment/env_WDTheory/bin/pip #points to the correct pip
#check what packages are already installed, save a list to a text file
>pip freeze --local > requirements.txt
#add to this text the packages that you want to install in the format <packageName>==<VersionNumber>, for example : pandas==1.4.4
#install packages one by one or all packages in the requirements.txt file
>pip install -r requirements.txt
#to deactivate the environment
>deactivate
