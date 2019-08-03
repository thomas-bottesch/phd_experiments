from __future__ import print_function
from experiment_db import ExperimentDB
from pprint import pprint
from collections import OrderedDict
import json
import pprint
import time
import numpy as np

def remove_where_kmeans_not_complete(result_data, ds, no_clusters, base_alg):
  remove_cluster_data = False
  
  if base_alg not in result_data:
    print("base_alg not in result_data", base_alg, result_data.keys(), remove_cluster_data)
    remove_cluster_data = True
  else:
      reference_run_set = set([0,1,2])
      if set(result_data[base_alg]['duration'].keys()) != reference_run_set:
        print("set(result_data[base_alg]['duration'].keys()) != reference_run_set", base_alg, result_data.keys(), remove_cluster_data)
        remove_cluster_data = True
      else:
        algs_to_delete = []
        reference_no_iterations = None
        for alg in result_data:
          if alg == base_alg:
            continue
          
          # verify that all runs were made
          if set(result_data[alg]['duration'].keys()) != reference_run_set:
            algs_to_delete.append(alg)
            continue
            
          """
          # verify that all algorithms had exactly the same number of iterations
          if reference_no_iterations is None:
            reference_no_iterations = result_data[alg]['no_iterations']
          
          if reference_no_iterations != result_data[alg]['no_iterations']:
            pprint(result_data)
            raise Exception("wrong iteration count")
          """

        for alg in algs_to_delete:
          del result_data[alg]
        
        if len(result_data) == 1:
          # only base_alg left
          print("len(result_data[alg]) == 1", base_alg, result_data.keys(), remove_cluster_data)
          remove_cluster_data = True
        else:
          
          for run in reference_run_set:
            kmeans_duration = result_data[base_alg]['duration'][run][0]

            """
            if result_data[base_alg]['no_iterations'][run] != reference_no_iterations[run]:
              print("need to approximate kmeans++ ds=%s, no_clusters=%s, run=%s, iterations=%s"
                    % (ds, str(no_clusters), str(run), str(result_data[base_alg]['no_iterations'][run])))
              iterations_done = len(result_data[base_alg]['iteration_durations'][run][0])
              kmeans_duration_per_iter = kmeans_duration / iterations_done
              kmeans_dur =  kmeans_duration_per_iter * reference_no_iterations[run]

              last_half_of_iterations = result_data[base_alg]['iteration_durations'][run][0][iterations_done / 2:]
              while len(result_data[base_alg]['iteration_durations'][run][0]) < reference_no_iterations[run]:
                result_data[base_alg]['iteration_durations'][run][0].append(np.average(last_half_of_iterations))
              
            else:
              kmeans_dur = kmeans_duration
            """

            kmeans_dur = kmeans_duration
            
            for alg in result_data:
              if alg == base_alg:
                continue
              
              if "bv" in alg or "pca" in alg:
                for param in result_data[alg]['duration'][run]:
                  alg_dur = result_data[alg]['duration'][run][param]
                  result_data[alg]['duration'][run][param] = kmeans_dur / alg_dur
                  print(alg, ds, no_clusters, run, param, result_data[alg]['duration'][run][param])
              else:
                alg_dur = result_data[alg]['duration'][run][0]
                result_data[alg]['duration'][run] = {0: kmeans_dur / alg_dur}
            
            result_data[base_alg]['duration'][run] = {0: 1.0}
  
  
  return remove_cluster_data

def remove_incomplete_data(result_data):
  
  datasets_to_delete = []
  
  for ds in result_data:
    clusters_to_delete = []
    for no_clusters in result_data[ds]['results']:
      if remove_where_kmeans_not_complete(result_data[ds]['results'][no_clusters], ds, no_clusters, 'kmeans++'):
        print("Remove incomplete data", ds, no_clusters)
        clusters_to_delete.append(no_clusters)
    
    for no_clusters in clusters_to_delete:
      del result_data[ds]['results'][no_clusters]
    
    if len(result_data[ds]['results']) == 0:
      datasets_to_delete.append(ds)
  
  for ds in datasets_to_delete:
    del result_data[ds]

def result_evaluation_kmeanspp(out_folder, out_folder_csv, remove_incomplete=False, ignore_datasets={}):
  
  for fcnt, plotname in [('do_kmeanspp', 'kmeans_params')]:
    run_identifiers = ExperimentDB.get_identifiers(out_folder, fcnt)
    plot_data = OrderedDict()
    
    result_data = OrderedDict()
    for run_identifier in run_identifiers:
      db = ExperimentDB(out_folder, fcnt, run_identifier)
      for resid in db.get_algorithm_run_ids():
        (control_params, params, res) = db.get_experiment_result_from_run_id(resid)
        if res is None:
          continue

        ds = params['info']['dataset_name']
        alg = params['info']['algorithm']
        no_clusters = params['task']['no_clusters']
        run = params['task']['run']
        duration_kmeans = res['duration_kmeans']
        #no_iterations = len(res['iteration_changes'])
        #iteration_durations = res['iteration_durations']
        #iteration_changes = res['iteration_changes']
        #iteration_wcssd = res['iteration_wcssd']
        
        if 'pca' in alg:
          param_percent = params['info']['truncated_svd_annz_percentage']
        elif 'bv' in alg:
          param_percent = params['task']['bv_annz']
        else:
          param_percent = 0
        
        if ds in ignore_datasets:
          continue

        if ds not in result_data:
          result_data[ds] = OrderedDict()
          result_data[ds]['results'] = OrderedDict()
          result_data[ds]['infos'] = OrderedDict()
          
        if no_clusters not in result_data[ds]['results']:
          result_data[ds]['results'][no_clusters] = OrderedDict()
          
        if alg not in result_data[ds]['results'][no_clusters]:
          result_data[ds]['results'][no_clusters][alg] = OrderedDict()
              
        for descr in ['duration']:
          if descr not in result_data[ds]['results'][no_clusters][alg]:
            result_data[ds]['results'][no_clusters][alg][descr] = OrderedDict()

        for descr in ['duration']:
          if run not in result_data[ds]['results'][no_clusters][alg][descr]:
            result_data[ds]['results'][no_clusters][alg][descr][run] = OrderedDict()

        kmeans_duration_this_run = duration_kmeans      
          
        if 'truncated_svd' in res:
          kmeans_duration_this_run += res['truncated_svd']['duration']

        if param_percent in result_data[ds]['results'][no_clusters][alg]['duration'][run]:
          raise Exception("dataset=%s no_clusters=%s alg=%s duration run=%s already added !!! %s %s" % (ds, str(no_clusters), alg, str(run), control_params, params))

        result_data[ds]['results'][no_clusters][alg]['duration'][run][param_percent] = kmeans_duration_this_run
        #result_data[ds]['results'][no_clusters][alg]['mem_consumption'][run][param_percent] = calculate_memory_consumption(alg, res)


        result_data[ds]['infos']['input_dimension'] = res['input_dimension']
        result_data[ds]['infos']['input_samples'] = res['input_samples']
        result_data[ds]['infos']['input_annz'] = res['input_annz']

        """"
        if run in result_data[ds]['results'][no_clusters][alg]['no_iterations']:
          if result_data[ds]['results'][no_clusters][alg]['no_iterations'][run] != no_iterations:
            print(alg, run, no_iterations, result_data[ds]['results'][no_clusters][alg]['no_iterations'][run], ds, no_clusters, param_percent, resid)
            raise Exception("Number of iterations is not identical!")
        else:
          result_data[ds]['results'][no_clusters][alg]['no_iterations'][run] = no_iterations
        

        result_data[ds]['results'][no_clusters][alg]['iteration_durations'][run][param_percent] = iteration_durations
        result_data[ds]['results'][no_clusters][alg]['iteration_changes'][run][param_percent] = iteration_changes
        result_data[ds]['results'][no_clusters][alg]['iteration_wcssd'][run][param_percent] = iteration_changes
        """
        
      if remove_incomplete:
        remove_incomplete_data(result_data)
    print(result_data)
    return result_data
