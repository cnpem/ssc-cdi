import subprocess
from subprocess import Popen, PIPE, STDOUT
import paramiko
import getpass
import time

madalda_ip = "10.30.4.10" # Mafalda IP
mafalda_port = 22

def connect_server():
    host = madalda_ip #"10.30.4.10" # Mafalda IP
    port = mafalda_port #22
    username = input("Username:")
    print("Password:")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, getpass.getpass())
    return ssh

def call_and_read_terminal(cmd,mafalda):
    if 0:
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        terminal_output = p.stdout.read() # Read output from terminal
    else:
        stdin, stdout, stderr = mafalda.exec_command(cmd)
        terminal_output = stdout.read() 
        print('Output: ',terminal_output)
        print('Error:  ',stderr.read())
    return terminal_output

def call_cmd_terminal(filename,mafalda,remove=False):
    cmd = f'sbatch {filename}'
    terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
    given_jobID = terminal_output.rsplit("\n",1)[0].rsplit(" ",1)[1]
    if remove: # Remove file after call
        cmd = f'rm {filename}'
        subprocess.call(cmd, shell=True)
        
    return given_jobID
        
def monitor_job_execution(given_jobID,mafalda):
    sleep_time = 10 # seconds
    print(f'Starting job #{given_jobID}...')
    time.sleep(3) # sleep for a few seconds to wait for job to really start
    jobDone = False
    job_duration = 0
    while jobDone == False:
        time.sleep(sleep_time)
        job_duration += sleep_time
        cmd = f'squeue | grep {given_jobID}'
        terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
        if given_jobID not in terminal_output:
            jobDone = True
        else:
            print(f'\tWaiting for job {given_jobID} to finish. Current duration: {job_duration/60:.2f} minutes')
    return print(f"\t \t Job {given_jobID} done!")

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
