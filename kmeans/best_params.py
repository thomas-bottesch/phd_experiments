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
    remove_cluster_data = True
  else:
      reference_run_set = set([0,1,2])
      if set(result_data[base_alg]['duration'].keys()) != reference_run_set:
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
            
          # verify that all algorithms had exactly the same number of iterations
          if reference_no_iterations is None:
            reference_no_iterations = result_data[alg]['no_iterations']
          
          if reference_no_iterations != result_data[alg]['no_iterations']:
            pprint(result_data)
            raise Exception("wrong iteration count")
        
        for alg in algs_to_delete:
          del result_data[alg]
        
        if reference_no_iterations is None:
          # only base_alg left
          remove_cluster_data = True
        else:
          
          for run in reference_no_iterations:
            kmeans_duration = result_data[base_alg]['duration'][run][0]
            if result_data[base_alg]['no_iterations'][run] != reference_no_iterations[run]:
              print("need to approximate kmeans ds=%s, no_clusters=%s, run=%s, iterations=%s"
                    % (ds, str(no_clusters), str(run), str(result_data[base_alg]['no_iterations'][run])))
              iterations_done = len(result_data[base_alg]['iteration_durations'][run][0])
              kmeans_duration_per_iter = kmeans_duration / iterations_done
              kmeans_dur =  kmeans_duration_per_iter * reference_no_iterations[run]

              last_half_of_iterations = result_data[base_alg]['iteration_durations'][run][0][iterations_done / 2:]
              while len(result_data[base_alg]['iteration_durations'][run][0]) < reference_no_iterations[run]:
                result_data[base_alg]['iteration_durations'][run][0].append(np.average(last_half_of_iterations))

            else:
              kmeans_dur = kmeans_duration
            
            for alg in result_data:
              if alg == base_alg:
                continue
              
              if "bv" in alg or "pca" in alg:
                for param in result_data[alg]['duration'][run]:
                  alg_dur = result_data[alg]['duration'][run][param]
                  result_data[alg]['duration'][run][param] = kmeans_dur / alg_dur
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
      if remove_where_kmeans_not_complete(result_data[ds]['results'][no_clusters], ds, no_clusters, 'kmeans'):
        print("Remove incomplete data", ds, no_clusters)
        clusters_to_delete.append(no_clusters)
    
    for no_clusters in clusters_to_delete:
      del result_data[ds]['results'][no_clusters]
    
    if len(result_data[ds]['results']) == 0:
      datasets_to_delete.append(ds)
  
  for ds in datasets_to_delete:
    del result_data[ds]


def calculate_memory_consumption(alg, res, ds, no_clusters, run, param):
  no_samples = res['input_samples']
  size_of_data_storage_element = 8
  size_of_key_storage_element = 4
  size_of_pointer_storage_element = 8
  
  if alg != 'kmeans':
    no_clusters_remaining = res['no_clusters_remaining']
  
  matrix_samples_mem_consumption = (res['input_annz'] * no_samples
                      * (size_of_data_storage_element + size_of_key_storage_element)) \
                      + ((no_samples + 1) * size_of_pointer_storage_element)
                      
  matrix_clusters_mem_consumption = (res['iteration_clusters_nnz'][-1]
                      * (size_of_data_storage_element + size_of_key_storage_element)) \
                      + ((no_samples + 1) * size_of_pointer_storage_element)

  kmeans_mem_consumption = float(matrix_samples_mem_consumption) + float(matrix_clusters_mem_consumption)
  if 'elkan' in alg:
    # Elkan mem consumption
    lower_bound_matrix_mem_consumption = no_samples * no_clusters_remaining * size_of_data_storage_element
    distance_between_clusters_matrix_mem_consumption = no_clusters_remaining * no_clusters_remaining * size_of_data_storage_element

  if "pca_" in alg:
    # PCA mem consumption
    orthonormal_basis_matrix_mem_consumption = (res['truncated_svd']['no_components']
                                                * res['truncated_svd']['no_features']
                                                * (size_of_data_storage_element + size_of_key_storage_element)) \
                                                + ((res['truncated_svd']['no_components'] + 1) * size_of_pointer_storage_element)
    pca_projected_matrix_samples_mem_consumption = (no_samples * res['truncated_svd']['no_components']
                        * (size_of_data_storage_element + size_of_key_storage_element)) \
                        + ((no_samples + 1) * size_of_pointer_storage_element)
                        
    pca_projected_matrix_clusters_mem_consumption = (no_clusters_remaining * res['truncated_svd']['no_components']
                        * (size_of_data_storage_element + size_of_key_storage_element)) \
                        + ((no_samples + 1) * size_of_pointer_storage_element)

  if 'bv_' in alg:
    # BV mem consumption
    annz_projected_matrix_samples = res['block_vector_data']['annz']
    # annz_projected_matrix_clusters was not measured (we use the annz_projected_matrix_samples as an approximation)
    annz_projected_matrix_clusters = annz_projected_matrix_samples
    
    bv_projected_matrix_samples_mem_consumption = (annz_projected_matrix_samples * no_samples
                        * (size_of_data_storage_element + size_of_key_storage_element)) \
                        + ((no_samples + 1) * size_of_pointer_storage_element)
                        
    bv_projected_matrix_clusters_mem_consumption = (annz_projected_matrix_clusters * no_clusters_remaining
                        * (size_of_data_storage_element + size_of_key_storage_element)) \
                        + ((no_samples + 1) * size_of_pointer_storage_element)

  if 'yinyang' in alg:
    # yinyang mem consumption
    t = no_clusters_remaining / 10
    lower_bound_group_matrix_mem_consumption = no_samples * t * size_of_data_storage_element

  if alg == 'kmeans':
    mem_consumption = 0
  elif alg == 'elkan':
    # elkan stores two dense matrices
    # 1. lower_bound_matrix = no_samples * no_clusters_remaining
    # 2. distance_between_clusters_matrix = no_clusters_remaining * no_clusters_remaining
    mem_consumption = lower_bound_matrix_mem_consumption + distance_between_clusters_matrix_mem_consumption

  elif alg == 'pca_elkan':  
    # pca_elkan stores two dense matrices + orthonormal_basis_matrix + projected_matrix_samples + projected_matrix_clusters
    # 1. lower_bound_matrix = no_samples * no_clusters_remaining
    # 2. distance_between_clusters_matrix = no_clusters_remaining * no_clusters_remaining
    # 3. orthonormal_basis_matrix = no_orthonormal_vectors * orthonormal_basis_matrix_dim
    # 4. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 5  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
    mem_consumption = lower_bound_matrix_mem_consumption \
                                                + distance_between_clusters_matrix_mem_consumption \
                                                + orthonormal_basis_matrix_mem_consumption \
                                                + pca_projected_matrix_samples_mem_consumption \
                                                + pca_projected_matrix_clusters_mem_consumption

  elif alg == 'bv_elkan':  
    # bv_elkan stores two dense matrices + orthonormal_basis_matrix + projected_matrix_samples + projected_matrix_clusters
    # 1. lower_bound_matrix = no_samples * no_clusters_remaining
    # 2. distance_between_clusters_matrix = no_clusters_remaining * no_clusters_remaining
    # 3. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 4. projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
    mem_consumption = lower_bound_matrix_mem_consumption \
                                                + distance_between_clusters_matrix_mem_consumption \
                                                + bv_projected_matrix_samples_mem_consumption \
                                                + bv_projected_matrix_clusters_mem_consumption

  elif alg == 'pca_kmeans':
    # pca_elkan stores a orthonormal_basis_matrix + projected_matrix
    # 1. orthonormal_basis_matrix = no_orthonormal_vectors * orthonormal_basis_matrix_dim
    # 2. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 3  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
    mem_consumption = orthonormal_basis_matrix_mem_consumption \
                      + pca_projected_matrix_samples_mem_consumption \
                      + pca_projected_matrix_clusters_mem_consumption
    
  elif alg == 'bv_kmeans':
    # bv_kmeans stores a projected_matrix_samples + projected_matrix_clusters
    # 1. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 2  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors
    mem_consumption = bv_projected_matrix_samples_mem_consumption \
                      + bv_projected_matrix_clusters_mem_consumption

  elif alg == 'yinyang':
    # yinyang stores a dense matrix to keep a lower bound to every of the t groups
    mem_consumption = lower_bound_group_matrix_mem_consumption
    
  elif alg == 'bv_yinyang':
    # yinyang stores a dense matrix to keep a lower bound to every of the t groups + block vector projected matrices samples/clusters
    # 1. lower_bound_group_matrix = no_samples * t
    # 2. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 3. projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
    mem_consumption = lower_bound_group_matrix_mem_consumption \
                      + bv_projected_matrix_samples_mem_consumption \
                      + bv_projected_matrix_clusters_mem_consumption

  elif alg == 'pca_yinyang':
    # yinyang stores a dense matrix to keep a lower bound to every of the t groups + block vector projected matrices samples/clusters
    # 1. lower_bound_group_matrix = no_samples * t
    # 2. orthonormal_basis_matrix = no_orthonormal_vectors * orthonormal_basis_matrix_dim
    # 3. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
    # 4  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
    mem_consumption = lower_bound_group_matrix_mem_consumption \
                      + pca_projected_matrix_samples_mem_consumption \
                      + orthonormal_basis_matrix_mem_consumption \
                      + pca_projected_matrix_clusters_mem_consumption
  elif alg == 'nc_kmeans':
    # nc_kmeans only consists of one array with an entry for every
    # sample saying if the sample is eligible for the optimization in that interation or not
    # a sample is only eligible for the optimization if its closest cluster stayed the same or moved towards it in
    # the last iteration. Then we do not have to look at the clusters that did not move.
    mem_consumption = no_samples * size_of_data_storage_element
  else:
    raise Exception("please provide details for the memory consumption of %s" % alg)

  if ds == 'e2006' and alg == 'bv_kmeans' and param == 0.25:
    print(ds, alg, no_clusters_remaining, run, bv_projected_matrix_samples_mem_consumption,
          bv_projected_matrix_clusters_mem_consumption, matrix_samples_mem_consumption,
          matrix_clusters_mem_consumption, (mem_consumption + kmeans_mem_consumption) / kmeans_mem_consumption)

  if ds == 'e2006' and alg == 'pca_kmeans':
    print(ds, alg, param, no_clusters_remaining, run, orthonormal_basis_matrix_mem_consumption,
          pca_projected_matrix_samples_mem_consumption, pca_projected_matrix_clusters_mem_consumption, matrix_samples_mem_consumption,
          matrix_clusters_mem_consumption, (mem_consumption + kmeans_mem_consumption) / kmeans_mem_consumption)

  return (mem_consumption + kmeans_mem_consumption) / kmeans_mem_consumption


def get_durations_create_blockvector(db, ignore_datasets={}):
  durations_create_bv = {}
  for resid in db.get_algorithm_run_ids():
    (control_params, params, res) = db.get_experiment_result_from_run_id(resid)
    if res is None:
      continue

    ds = params['info']['dataset_name']
    alg = params['info']['algorithm']
    no_clusters = params['task']['no_clusters']
    run = params['task']['run']
    duration_kmeans = res['duration_kmeans']
    duration_init = res['duration_init']
    iteration_durations = res['iteration_durations']

    if 'pca' in alg:
      param_percent = params['info']['truncated_svd_annz_percentage']
    elif 'bv' in alg:
      param_percent = params['task']['bv_annz']
    else:
      param_percent = 0
    
    if ds in ignore_datasets:
      continue

    if 'bv' in alg:
      if (param_percent == 0.25 and alg =="bv_kmeans"):
        if ds not in durations_create_bv:
          durations_create_bv[ds] = {}
        
        if no_clusters not in durations_create_bv[ds]:
          durations_create_bv[ds][no_clusters] = []

        durations_create_bv[ds][no_clusters].append(duration_kmeans - (sum(iteration_durations) + duration_init))
  
  for ds in durations_create_bv:
    for no_clusters in durations_create_bv[ds]:
      durations_create_bv[ds][no_clusters] = sum(durations_create_bv[ds][no_clusters]) / len(durations_create_bv[ds][no_clusters])

  return durations_create_bv

def result_evaluation_best_params(out_folder, out_folder_csv, remove_incomplete=False, ignore_datasets={}):
  
  fcnt, plotname = ('do_best_params', 'kmeans_params')
  run_identifiers = ExperimentDB.get_identifiers(out_folder, fcnt)
  plot_data = OrderedDict()
  
  impact_all = {'bv': {}, 'pca': {}}

  result_data = OrderedDict()
  for run_identifier in run_identifiers:
    db = ExperimentDB(out_folder, fcnt, run_identifier)
    durations_create_bv = get_durations_create_blockvector(db, ignore_datasets=ignore_datasets)
    for resid in db.get_algorithm_run_ids():
      (control_params, params, res) = db.get_experiment_result_from_run_id(resid)
      if res is None:
        continue

      ds = params['info']['dataset_name']
      alg = params['info']['algorithm']
      no_clusters = params['task']['no_clusters']
      run = params['task']['run']
      duration_kmeans = res['duration_kmeans']
      duration_init = res['duration_init']
      no_iterations = len(res['iteration_changes'])
      iteration_durations = res['iteration_durations']
      iteration_changes = res['iteration_changes']
      iteration_wcssd = res['iteration_wcssd']
       
      if 'pca' in alg:
        param_percent = params['info']['truncated_svd_annz_percentage']
        impact = impact_all['pca']
      elif 'bv' in alg:
        param_percent = params['task']['bv_annz']
        impact = impact_all['bv']
      else:
        param_percent = 0
      
      if ds in ignore_datasets:
        continue

      #if alg == "pca_elkan" and no_clusters == 100 and ds == "mediamill":
      #  continue

      if ds not in result_data:
        result_data[ds] = OrderedDict()
        result_data[ds]['results'] = OrderedDict()
        result_data[ds]['infos'] = OrderedDict()
        
      if no_clusters not in result_data[ds]['results']:
        result_data[ds]['results'][no_clusters] = OrderedDict()
        
      if alg not in result_data[ds]['results'][no_clusters]:
        result_data[ds]['results'][no_clusters][alg] = OrderedDict()
            
      for descr in ['iteration_durations', 'iteration_changes', 'iteration_wcssd', 'duration', 'mem_consumption', 'no_iterations']:
        if descr not in result_data[ds]['results'][no_clusters][alg]:
          result_data[ds]['results'][no_clusters][alg][descr] = OrderedDict()

      for descr in ['iteration_durations', 'iteration_changes', 'iteration_wcssd', 'duration', 'mem_consumption']:
        if run not in result_data[ds]['results'][no_clusters][alg][descr]:
          result_data[ds]['results'][no_clusters][alg][descr][run] = OrderedDict()

      kmeans_duration_this_run = duration_kmeans      
        
      if 'truncated_svd' in res:
        kmeans_duration_this_run += res['truncated_svd']['duration']
        if (param_percent == 0.06):
          if no_clusters not in impact:
            impact[no_clusters] = []
          impact[no_clusters].append(res['truncated_svd']['duration'] / (duration_kmeans + res['truncated_svd']['duration']))
          print(ds, alg, no_clusters, run, "%.3f" % (res['truncated_svd']['duration'] / (duration_kmeans + res['truncated_svd']['duration']) ))
      
      if 'bv' in alg:
        if (param_percent == 0.25):
          if no_clusters not in impact:
            impact[no_clusters] = []
          impact[no_clusters].append(durations_create_bv[ds][no_clusters] / duration_kmeans)
          print(ds, alg, no_clusters, run, "%.3f" % (durations_create_bv[ds][no_clusters] / duration_kmeans))

      if param_percent in result_data[ds]['results'][no_clusters][alg]['duration'][run]:
        raise Exception("dataset=%s no_clusters=%s alg=%s duration run=%s already added !!! %s %s" % (ds, str(no_clusters), alg, str(run), control_params, params))

      result_data[ds]['results'][no_clusters][alg]['duration'][run][param_percent] = kmeans_duration_this_run
      result_data[ds]['results'][no_clusters][alg]['mem_consumption'][run][param_percent] = calculate_memory_consumption(alg, res, ds, no_clusters, run, param_percent)


      result_data[ds]['infos']['input_dimension'] = res['input_dimension']
      result_data[ds]['infos']['input_samples'] = res['input_samples']
      result_data[ds]['infos']['input_annz'] = res['input_annz']

      if run in result_data[ds]['results'][no_clusters][alg]['no_iterations']:
        if result_data[ds]['results'][no_clusters][alg]['no_iterations'][run] != no_iterations:
          print(alg, run, no_iterations, result_data[ds]['results'][no_clusters][alg]['no_iterations'][run], ds, no_clusters, param_percent, resid)
          raise Exception("Number of iterations is not identical! len(res['iteration_changes']) = %d, no_iterations= %d resid=%d" % (no_iterations, result_data[ds]['results'][no_clusters][alg]['no_iterations'][run], resid))
      else:
        result_data[ds]['results'][no_clusters][alg]['no_iterations'][run] = no_iterations
      

      result_data[ds]['results'][no_clusters][alg]['iteration_durations'][run][param_percent] = iteration_durations
      result_data[ds]['results'][no_clusters][alg]['iteration_changes'][run][param_percent] = iteration_changes
      result_data[ds]['results'][no_clusters][alg]['iteration_wcssd'][run][param_percent] = iteration_changes
      
    if remove_incomplete:
      remove_incomplete_data(result_data)
    
    for name in impact_all:
      print("Impact of the creation of %s data structures on the total running time" % name)
      for no_clusters in sorted(impact_all[name]):
        lst = impact_all[name][no_clusters]
        print('{:>5} {:>6} {:>6}'.format(str(no_clusters),
                                           "%.3f" % (sum(lst) / len(lst)),
                                           "%.3f" % np.max(lst)))

    return result_data
