import subprocess

def install_packages(mafalda):
    print('Installing packages @ Bertha (local)...')
    cmd = """pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
pip config --user set global.trusted-host gcc.lnls.br
pip install sscCdi==0.0.5
pip install sscPimega==0.0.4
pip install sscRaft==1.0.0
pip install sscResolution==1.2.3"""
    for line in cmd.split('\n'):
        print('\t',line)
        subprocess.call(line, shell=True)
    print('\t Done!')

    # print('Installing packages @ Mafalda (cluster)...')
    # cmd = """module load python3/3.9.2
    # module load cuda/11.2
    # module load hdf5/1.12.0_parallel
    # pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
    # pip config --user set global.trusted-host gcc.lnls.br
    # pip install sscCdi==0.0.5
    # pip install sscPimega==0.0.4
    # pip install sscRaft==1.0.2
    # pip install sscResolution==1.2.3"""
    # for line in cmd.split('\n'):
    #     stdin, stdout, stderr = mafalda.exec_command(line)
    #     print('Output: ',stdout.read())
    #     print('Error:  ',stderr.read())   
    # print('\t Done!')
