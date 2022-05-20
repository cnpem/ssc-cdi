from git import Repo
import os

def create_directory_if_doesnt_exist(*args):
    for arg in args:
        if check_if_folder_exists(folder): == False:
            os.mkdir(arg)
            
def check_if_folder_exists(folder):
	if os.path.isdir(folder) == False:
	    return False
	else:
	    return True
            
# Repositories:
repositories = [['https://gitlab.cnpem.br/GCC/ssc-cdi.git'                , True,  557d1351063775a22c345fadc8b20d4fdc8b9235],
		['https://gitlab.cnpem.br/GCC/ssc-pimega.git'              , False, c7255336a9aabfb73d0c960d5c87bbdf44dc5379],
		['https://gitlab.cnpem.br/GCC/ssc-raft.git'                , True,  8309eb22696afd436522a188c4615ab2d052471f],
		['https://gitlab.cnpem.br/GCC/reconstruction/ssc-radon.git', True,  9571f5cc31d8ff92fa9f0e06cb05cdc51bdca8b3]]
# .whl sscPtycho
# sscIo https://gitlab.cnpem.br/GCC/sscIO.git

errors_list = []
for counter,repository in enumarate(repositories):

	# Clone repositories
	git_url  = repository[0]
	repo_dir = os.path.join('~/',git_url.rsplit('/',1)[1][:-4])
	
	if check_if_folder_exists(repo_dir) == True:
		#TODO check if desired commit, otherwise, continue
		# git rev-parse HEAD
		errors_list.append(f'Desired package and commit already installed: {repo_dir, repository[2]}')
		continue
		
	create_directory_if_doesnt_exist(repo_dir) # create folder for cloning, if needed
	repositories[counter].append(repo_dir)
	Repo.clone_from(git_url, repo_dir) # clone repository
	
	#TODO Checkout commit of interest
	#TODO Go to repo dir
	
	if repository[1] == True: # check if --cuda is needed
		install_string = 'python3 setup.py install --user --cuda'
	else:
		install_string = 'python3 setup.py install --user'
	
	#TODO Install and check if succesfull:
	# try:
	#	errors_list.append(f'Installed {repo_dir} successfully!')
	# except:
	#	errors_list.append(f'Could NOT install {repo_dir}! ')
	
	#TODO delete repositories
		
print('Finished installing packages!)
print('Results:')
for error in errors_list:
	print(error)		
		
		
		
		
		
	
