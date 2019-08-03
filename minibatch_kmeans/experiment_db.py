import os
import cPickle
import json
import time
from os.path import abspath, join, dirname

class ExperimentDB():
  
  def __init__(self, out_folder, function_name, run_identifier, dump_also_as_json=False):
    self.output_folder_local = join(out_folder, function_name)
    self.run_identifier = run_identifier
    self.dump_also_as_json = dump_also_as_json
    if not os.path.isdir( self.output_folder_local):
      os.makedirs(self.output_folder_local)
    
    self.output_file_path_database = join(self.output_folder_local, "index" + "_" + self.run_identifier)
    if os.path.isfile(self.output_file_path_database):
      with open(self.output_file_path_database, 'rb') as f:
        self.db = json.load(f)
        self.next_algorithm_run_id = self.get_last_algorithm_run_id() + 1
    else:
      self.db = []
      self.next_algorithm_run_id = 0
  
  def get_ikeys(self):
    return [int(k) for k in self.db.keys()]
  
  def get_algorithm_run_ids(self):
    return [self.db[k][0] for k in range(len(self.db))]
  
  @classmethod
  def get_identifiers(cls, out_folder, function_name):
    run_identifiers = []
    output_folder_local = join(out_folder, function_name)
    for f in os.listdir(output_folder_local):
      if f.startswith("index_"):
        if "~" in f:
          continue
        run_identifier = f.replace("index_", "")
        run_identifiers.append(run_identifier)
    return run_identifiers
  
  def get_last_experiment_result_only(self):
    if len(self.db) == 0:
      return None
    else:
      (control_params, params, res) = self.get_experiment_result_from_run_id(self.db[-1][0])
      return res
  
  def get_last_algorithm_run_id(self):
    last_number = 0
    for f in os.listdir(self.output_folder_local):
      if f.startswith(self.run_identifier + "_") and not f.endswith(".txt"):
        runid = int(f.replace(self.run_identifier + "_", ""))
        if runid > last_number:
          last_number = runid
      
    return last_number
  # retrieves the experiment result for a set of parameters if it exists
  # returns None if for this set of parameters no experiment was done
  def get_experiment_result(self, params):
    previous_runs = [algorithm_run_id for algorithm_run_id, prms in self.db if params == prms]
    if len(previous_runs) != 0:
      algorithm_run_id = previous_runs[0]
    else:
      return None
    
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(algorithm_run_id)) 
    with open(output_file_run, 'rb') as f:
      return cPickle.load(f)
    
  # retrieves the experiment result for a set of parameters if it exists
  # returns None if for this set of parameters no experiment was done
  def get_experiment_result_from_run_id(self, algorithm_run_id):
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(algorithm_run_id)) 
    with open(output_file_run, 'rb') as f:
      return cPickle.load(f)
  
  def remove_experiments_with_run_ids(self, delete_run_ids):
    if delete_run_ids is None:
      return
    if type(delete_run_ids) != dict:
      return
    if len(delete_run_ids) == 0:
      return
    del_id = None
    new_db = []
    for i in range(len(self.db)):
      (run_id, prms) = self.db[i]
      if run_id not in delete_run_ids:
        new_db.append(self.db[i])
    
    self.persist_db(with_backup=new_db)
      
  def persist_db(self, with_backup=None):
    
    if with_backup is not None:
      backup_output_file_path_database = join(self.output_folder_local,
                                              "backup-index" + "_" + self.run_identifier + "-" + str(time.time()))
      with open(backup_output_file_path_database, 'wb') as f:
        json.dump(self.db, f, indent=4)
      self.db = with_backup
      
    with open(self.output_file_path_database, 'wb') as f:
      json.dump(self.db, f, indent=4)
  
  def add_experiment(self, control_params, params, res):
    
    while True: 
      output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(self.next_algorithm_run_id))
      if not os.path.isfile(output_file_run):
        break
      self.next_algorithm_run_id += 1
    
    self.db.append((self.next_algorithm_run_id,params))
    
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(self.next_algorithm_run_id))
    with open(output_file_run, 'wb') as f:
      cPickle.dump((control_params, params, res), f)
    
    if self.dump_also_as_json:
      output_file_run_json = output_file_run + ".txt"
      with open(output_file_run_json, 'wb') as f:
        json.dump(res, f, indent=4)
      
    self.next_algorithm_run_id += 1
    self.persist_db()

def delete_10k_iteration_limit(output_path, function):
  run_identifiers = ExperimentDB.get_identifiers(output_path, function)
  for run_identifier in run_identifiers:
    db = ExperimentDB(output_path, function, run_identifier)
    ids_to_delete = {}
    for resid in db.get_algorithm_run_ids():
      (control_params, params, res) = db.get_experiment_result_from_run_id(resid)
      if params['task']['iteration_limit'] == 10000:
        ids_to_delete[resid] = None
    db.remove_experiments_with_run_ids(ids_to_delete)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Experiment to evaluate k-means variants with feature maps that exploit'
                                               ' different representations of data sets')
  parser.add_argument('--output_path', type=str, default="output_path", help='Path to the results of single algorithm executions')
  parser.add_argument('--function', type=str, default="do_minibatch_best_params", help='Path to the results of single algorithm executions')
  args = parser.parse_args()
  
  delete_10k_iteration_limit(args.output_path, args.function)