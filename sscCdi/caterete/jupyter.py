import subprocess
from subprocess import Popen, PIPE, STDOUT
import time, os
import paramiko
import getpass

def write_to_file(python_script_path,jsonFile_path,output_path="",slurmFile = 'slurmJob.sh',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    # Create slurm file
    string = f"""#!/bin/bash

#SBATCH -J {jobName}          # Select slurm job name
#SBATCH -p {queue}            # Fila (partition) a ser utilizada
#SBATCH --gres=gpu:{gpus}     # Number of GPUs to use
#SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
#SBATCH -o ./slurm.out        # Select output path of slurm file

module load python3/3.9.2
module load cuda/11.2
module load hdf5/1.12.0_parallel

# python3 {python_script_path} {jsonFile_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}
python3 {python_script_path} > {os.path.join(output_path,'output.log')} 2> {os.path.join(output_path,'error.log')}

"""
    
    with open(slurmFile,'w') as the_file:
        the_file.write(string)
    
    return slurmFile

def call_and_read_terminal(cmd,mafalda):
    if 0:
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        terminal_output = p.stdout.read() # Read output from terminal
    else:
        stdin, stdout, stderr = mafalda.exec_command(cmd)
    terminal_output = stdout.read() 
    # print('Output: ',terminal_output)
    # print('Error:  ',stderr.read())
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
    time.sleep(3) # sleep for a few seconds to wait for job to really start
    jobDone = False
    while jobDone == False:
        time.sleep(2)
        cmd = f'squeue | grep {given_jobID}'
        terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
        if given_jobID not in terminal_output:
            jobDone = True
        else:
            print(f'\tWaiting for job {given_jobID} to finish')
    return print(f"\t \t Job {given_jobID} done!")

def run_ptycho_from_jupyter(mafalda,python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32):
    
    # python_script_path = 'testpy.py'
    jsonFile_path = ''
    slurm_file = write_to_file(python_script_path,jsonFile_path,output_path,slurmFile,jobName,queue,gpus,cpus)
    given_jobID = call_cmd_terminal(slurm_file,mafalda,remove=False)
    monitor_job_execution(given_jobID,mafalda)

    
def connect_mafalda():
    host = "10.30.4.10" # Mafalda IP
    port = 22
    username = input("Username:")
    print("Password:")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, getpass.getpass())
    return ssh


if __name__ == "__main__":

    python_script_path = 'testpy.py'
    jsonFile_path = ''
    run_ptycho_from_jupyter(python_script_path,jsonFile_path,output_path="",slurmFile = 'ptychoJob2.srm',jobName='jobName',queue='cat-proc',gpus=1,cpus=32)