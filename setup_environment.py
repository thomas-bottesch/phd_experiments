#!/usr/bin/python
from os.path import isfile, isdir, ismount, expanduser, join, sep, basename, abspath, dirname
from os import makedirs
from subprocess import Popen
import argparse
import shlex
import sys
import os
import shutil
import time
import tarfile
from urllib.request import urlretrieve

if sys.version_info[0] < 3:
    print("This script needs to be executed with python3")
    sys.exit(-1)

home = expanduser("~")
downloads_folder = join(home, "Downloads")
miniconda_install_dir = join(home, "phd_exp_miniconda3")
miniconda_bin_dir = join(miniconda_install_dir, 'bin')
miniconda_executable = join(miniconda_bin_dir, 'conda')
venvs_folder = join(home, ".phd_exp_virtualenvs")
phd_exp_conda_env_path = join(venvs_folder, "phd_exp")
pip_env_executable = join(phd_exp_conda_env_path, 'bin', 'pip')
python_env_executable = join(diss_conda_env_path, 'bin', 'python')
directory_of_this_script =  abspath(dirname(__file__))
dataset_path = join(directory_of_this_script, "datasets")

def create_folders(folder_list):
    for folder in folder_list:
        if not isdir(folder):
            print("Creating folder %s" % folder)
            makedirs(folder)

def verify_exists_dir(d):
    if not isdir(d):
        raise Exception("Directory %s should exist but it does not!" % d)

def download_file_from_url(url, destination):
    if isfile(destination):
        print("%s already exists" % destination)
        return
    else:
        print("Downloading %s to %s" % (url, destination))

    from urllib.request import urlretrieve
    urlretrieve(url, destination)

def give_process_priority(pid):
    os.system("sudo renice -n -20 -p " + str(pid))

def execute_program(cmd, env=None, cwd=None, with_priority=False):

    if type(cmd) is list:
        cmd_list = cmd
    else:
        cmd_list = shlex.split(cmd)

    print("Executing \"%s\"" % " ".join(cmd_list))
    x = Popen(cmd_list, env=env, cwd=cwd)
    if with_priority:
        print("Setting priority of program \"%s\" to maximum. (Might require sudo)" % " ".join(cmd_list))
        give_process_priority(x.pid)
    
    x.wait()
    if x.returncode != 0:
        raise Exception("Error while executing %s. Returncode %d" % (str(cmd_list), x.returncode))

def install_latest_miniconda():
    if isdir(miniconda_install_dir):
        return
    else:
        print("Miniconda does not exist yet. Trying to install it!")

    filename = "Miniconda3-latest-Linux-x86_64.sh"
    download_destination = join(downloads_folder, filename)
    download_file_from_url("https://repo.anaconda.com/miniconda/%s" % filename, join(downloads_folder, filename))

    execute_program("bash %s -p %s -b" % (download_destination, miniconda_install_dir))
    print("Miniconda was installed successfully into %s" % miniconda_install_dir)
    os.remove(download_destination)

def install_phd_exp_conda_environment(force=False):
    
    if isdir(phd_exp_conda_env_path):
        if force:
            shutil.rmtree(phd_exp_conda_env_path)
        else:
            return
    else:
        print("The virtual environment to run the experiments does not exist yet. Trying to create it.")
    
    install_latest_miniconda()

    print("Installing some requirements via apt. (sudo access is required)")
    execute_program("sudo apt install libpng-dev libfreetype6-dev libx11-dev")

    execute_program("%s create -y python=2.7 -p %s" % (miniconda_executable, phd_exp_conda_env_path))

    packages_to_install = ["cycler==0.10.0", "Cython==0.24.1", "dask==1.1.1", "decorator==4.3.2",
                           "matplotlib==1.5.0", "networkx==2.2", "numexpr==2.5", "numpy==1.13.1",
                           "Pillow==5.4.1", "pyparsing==2.3.1", "python-dateutil==2.8.0",
                           "pytz==2018.9", "scikit-image==0.12.3", "scikit-learn==0.18.1",
                           "scipy==0.19.0", "six==1.12.0", "toolz==0.9.0", "fcl==0.0.12"]

    print("Installing some requirements via pip.")
    for pkg in packages_to_install:
        execute_program("%s install %s" % (pip_env_executable, pkg))
    
    print("The environment was successfully set up.")

def check_expected_args(args_dict, expected_arg_list):
    for el in expected_arg_list:
        if el not in args_dict:
            raise Exception("Expected argument %s not in args_dict" % str(el))

"""
  parser.add_argument('--dataset_folder', type=str, default="datasets", help='Path datasets are downloaded to')
  parser.add_argument('--output_path', type=str, default="output_path", help='Path to the results of single algorithm executions')
  parser.add_argument('--output_path_latex', type=str, default="output_path_latex", help='Path to the results as latex tables')
  parser.add_argument('--output_path_latex_append_folder', type=str, default="bestparams")
  parser.add_argument('--only_result_evaluation', dest='only_evaluation', action='store_true',
                      help='Recreate latex tables based on previous results without executing kmeans again')
  parser.add_argument('--testmode', dest='testmode', action='store_true', help='Only run the experiments for a single small dataset')
  parser.set_defaults(only_evaluation=False)
  parser.set_defaults(testmode=False)
  
"""


def execute_experiment(args):
    install_phd_exp_conda_environment()
    create_folders(dataset_path)

    print("Executing experiments for %s")
    experiments_script = join(directory_of_this_script, args["exp_type"], "do_experiments.py")
    execute_program("%s %s %s %s %s %s" % (python_env_executable,
                                     experiments_script,
                                     "--dataset_folder",
                                     dataset_path
                                     "--testmode" if args["testmode"] else "",
                                     "--only_result_evaluation" if args["only_result_evaluation"] else ""))

def create_subparser(subparsers, command_name, command_data):
    p = subparsers.add_parser(command_name, help=command_data.get('help', ''))
    command_data['parser'] = p
    args = command_data.get('args', None)
    if args is not None:
        for arg in args:
            if type(args[arg]) == dict:
                h=args[arg].get('help', '')
                command_data['parser'].add_argument(arg, help=h)
            else:
                # this is a regular argument
                continue
            
            
def execute_chosen_environment(parsed_args, command_data):
    pargs_dict = vars(parsed_args)

    args = command_data.get('args', None)
    if args is not None:
        for arg in args:
            if type(args[arg]) == dict:
                args[arg] = pargs_dict[arg]
            else:
                # this is a regular argument
                continue

    command_data.get('method')(args)

def uninstall(args):
    if isdir(phd_exp_conda_env_path):
        shutil.rmtree(phd_exp_conda_env_path)
    
    if isdir(venvs_folder):
        shutil.rmtree(venvs_folder)

def main():
    parser = argparse.ArgumentParser(prog='PROG')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    algorithms = ["kmeans", "kmeanspp", "minibatch_kmeans"]
    opts = {}

    opts["uninstall"] = {'method': uninstall,
                         'help': 'Remove everything related to the phd_experiments. The git repo will remain!'}

    for alg in algorithms:
        opts[alg] = {'method': execute_experiment,
                       'args': {"exp_type": alg},
                       'help': 'Execute the %s experiments. (This can take multiple weeks/month)' % alg}

        opts[alg + "_mini"] = {'method': execute_experiment,
                               'args': {"exp_type": alg, "testmode": True},
                               'help': 'Execute the %s experiments. (This can take multiple weeks/month)' % alg}

    for command in opts:
        create_subparser(subparsers, command, opts[command])

    parsed_args = parser.parse_args()
    execute_chosen_environment(parsed_args, opts[parsed_args.subparser_name])

if __name__ == "__main__":
    main()
