from __future__ import print_function
import argparse
import os
import collections
import traceback
import time
import copy
from experiment_db import ExperimentDB
from sklearn.decomposition import TruncatedSVD
from Queue import Empty
from multiprocessing import Process, Queue
from os.path import abspath, join, dirname
from collections import OrderedDict
from fcl import kmeans
from fcl.datasets import load_sector_dataset, load_usps_dataset, load_and_extract_dataset_from_github
from fcl.matrix.csr_matrix import get_csr_matrix_from_object, csr_matrix_to_libsvm_string
from best_params import result_evaluation_best_params
from latex_plots_best_params import create_plot
from latex_plots_iteration_durations import create_iteration_duration_plot
from latex_plots_table import create_table, TABLE_TYPE_DURATION, TABLE_TYPE_MEMORY

# params contains parameter which modify the internals of the funtion_to_evaluate
# if params get changed and do not exist in the database, funtion_to_evaluate is
# run for the params
# control_params contain data which is needed for function to evaluate 
# e.g. control_params contains {dataset_path: <..>, dataset_name: <..>, out_folder: <..>]}
# params contain:
# params_general['tsne'] = collections.OrderedDict()
# params_general['tsne']['tsne_components'] = "2"
# params_general['tsne']['perplexity'] = "30."
# params_general['tsne']['theta'] = "0.5"
# params_general['tsne']['algorithm'] = "bh_sne"
# a change in control params does not lead to a reevaluation
# run_identifier needed to determine the current database. often the dataset name is used 
def evaluate(run_identifier, control_params, params, function_to_evaluate, out_folder, force = False):
  try:
    experiment_db = ExperimentDB(out_folder, function_to_evaluate.__name__, run_identifier, dump_also_as_json=True)
    previous_result = experiment_db.get_experiment_result(params)
    
    if previous_result is not None and not force:
      print("already_exists:", function_to_evaluate.__name__ + " for", run_identifier, "with", params)
      return previous_result
    else:
      print(function_to_evaluate.__name__ + " for", run_identifier, "with", params)
      result_q = Queue()
      p = Process(target=function_to_evaluate, args=(result_q, control_params, params))
      p.start()
      p.join() # this blocks until the process terminates
      
      try:
        res = result_q.get_nowait()
      except Empty:
        print("no result available for this call. the process most likely failed with %s" % str(p.exitcode))
        res = None
      
      experiment_db.add_experiment(control_params, params, res)
      return (control_params, params, res)
  except:
    raise


def do_best_params(result_q, control_params, params):
    
    X = control_params['libsvm_dataset_path']
    
    data_as_csrmatrix = get_csr_matrix_from_object(X)
    no_samples, _ = data_as_csrmatrix.shape
    
    info = params['info']
    task = params['task']
    
    output = "for %s with algorithm=%s run=%d k=%d"%(info['dataset_name'],
                                                              info['algorithm'],
                                                              task['run'],
                                                              task['no_clusters'])
    
    if 'pca' in info['algorithm']:
      
      annz_input_matrix = data_as_csrmatrix.annz
      desired_no_eigenvectors = int(data_as_csrmatrix.annz * info['truncated_svd_annz_percentage'])
      print("Using TruncatedSVD to retrieve %d eigenvectors from input matrix with %d annz" % (desired_no_eigenvectors,
                                                                                               annz_input_matrix))
      p = TruncatedSVD(n_components = int(data_as_csrmatrix.annz * info['truncated_svd_annz_percentage']))
      start = time.time()
      scipy_csr_matrix = data_as_csrmatrix.to_numpy()
      p.fit(scipy_csr_matrix)
      # convert to millis
      fin = (time.time() - start) * 1000 
      pca_projection_csrmatrix = get_csr_matrix_from_object(p.components_)
      (no_components, no_features) = p.components_.shape
      print("Time needed to complete getting %d eigenvectors with %d features with SVD:" % (no_components, no_features),
            fin, "(annz of the top eigenvectors:", pca_projection_csrmatrix.annz, ")")
      additional_algo_data = {info['algorithm']: {'data': pca_projection_csrmatrix, 'duration': fin}}
    else:
      additional_algo_data = {}
    
    print("Executing " + output)
    km = kmeans.KMeans(n_jobs=1, no_clusters=task['no_clusters'], algorithm=info['algorithm'],
                       init='random', seed = task['run'], verbose = True, additional_params = dict(task),
                       iteration_limit = task['iteration_limit'], additional_info = dict(info))
    
    if info['algorithm'] in additional_algo_data:
      km.fit(data_as_csrmatrix, external_vectors = additional_algo_data[info['algorithm']]['data'])
      result = km.get_tracked_params()
      result['truncated_svd'] = {}
      result['truncated_svd']['no_components'] = no_components
      result['truncated_svd']['no_features'] = no_features
      result['truncated_svd']['duration'] = additional_algo_data[info['algorithm']]['duration']
    else:
      km.fit(data_as_csrmatrix)
      result = km.get_tracked_params()
    
    result_q.put(result)
        
def do_evaluations(dataset_path, dataset_name, out_folder, params, clusters, truncated_svd_annz_percentage, bv_annz):
  
  algorithms = ["kmeans", "bv_kmeans", "pca_kmeans", "elkan", "bv_elkan", "yinyang", "pca_elkan", 'bv_yinyang', 'pca_yinyang', "nc_kmeans"]
  
  control_params = OrderedDict()
  control_params['libsvm_dataset_path'] = dataset_path
  slow_algorithms = ["kmeans", "minibatch_kmeans", "kmeans++"]
  huge_datasets = ["caltech101", "kdd2001", "mnist800k"]

  if 'calculate_kmeans' in params:
    for no_clusters in clusters:
      for algorithm in algorithms:
        
        for r in range(3):
          params['calculate_kmeans']['task'] = collections.OrderedDict()
          params['calculate_kmeans']['task']['no_clusters'] = no_clusters
          params['calculate_kmeans']['task']['run'] = r
          if algorithm in slow_algorithms and dataset_name in huge_datasets:
            # for the standard algorithms we only do a few iterations in huge datasets
            # we then calculate the average duration of these three iterations 
            # and extrapolate to retrieve the time needed to calculate
            # the full kmeans
            params['calculate_kmeans']['task']['iteration_limit'] = 10
          else:
            params['calculate_kmeans']['task']['iteration_limit'] = 1000
                   
          params['calculate_kmeans']['info'] = collections.OrderedDict()
          params['calculate_kmeans']['info']['dataset_name'] = dataset_name
          params['calculate_kmeans']['info']['algorithm'] = algorithm
          
          params_specific = copy.deepcopy(params)
          
          if 'pca' in algorithm:
            for perc in truncated_svd_annz_percentage:
              params_specific['calculate_kmeans']['info']['truncated_svd_annz_percentage'] = perc
              
              try:
                evaluate("best_params", control_params, params_specific['calculate_kmeans'], do_best_params, out_folder)
              except:
                print(traceback.format_exc())
          elif 'bv' in algorithm:
            for perc in bv_annz:
              params_specific['calculate_kmeans']['task']['bv_annz'] = perc
              
              try:
                evaluate("best_params", control_params, params_specific['calculate_kmeans'], do_best_params, out_folder)
              except:
                print(traceback.format_exc())
          else:
            try:
              evaluate("best_params", control_params, params_specific['calculate_kmeans'], do_best_params, out_folder)
            except:
              print(traceback.format_exc())

def remove_dataset_from_params(result_data, ds_dict):
  
  for ds in ds_dict:
    if ds in result_data:
      del result_data[ds]
  return result_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment to evaluate k-means variants with feature maps that exploit'
                                               ' different representations of data sets')
  parser.add_argument('--dataset_folder', type=str, default="datasets", help='Path datasets are downloaded to')
  parser.add_argument('--output_path', type=str, default="output_path", help='Path to the results of single algorithm executions')
  parser.add_argument('--output_path_latex', type=str, default="output_path_latex", help='Path to the results as latex tables')
  parser.add_argument('--output_path_latex_append_folder', type=str, default="bestparams")
  parser.add_argument('--only_result_evaluation', dest='only_evaluation', action='store_true',
                      help='Recreate latex tables based on previous results without executing kmeans again')
  parser.add_argument('--testmode', dest='testmode', action='store_true', help='Only run the experiments for a single small dataset')
  parser.set_defaults(only_evaluation=False)
  parser.set_defaults(testmode=False)
  
  args = parser.parse_args()
  
  out_folder = abspath(args.output_path)
  output_path_latex = os.path.join(abspath(args.output_path_latex), args.output_path_latex_append_folder)
  if not args.only_evaluation:
    ds_folder = abspath(args.dataset_folder)
    
    if not os.path.isdir(ds_folder):
      os.makedirs(ds_folder)
    
    if not os.path.isdir(out_folder):
      os.makedirs(out_folder)
       
    params_general = collections.OrderedDict()
    params_general['calculate_kmeans'] = collections.OrderedDict()
    clusters_big = [100, 1000, 10000]
    clusters_medium = [100, 500, 5000]
    clusters_small = [100, 250, 1000]

    truncated_svd_annz_percentage = [float(x) / 100.0 for x in range(2, 42, 2)]
    bv_annz = [float(x) / 100.0 for x in range(5, 75, 5)]
    do_evaluations(load_usps_dataset(ds_folder), 'usps', out_folder, params_general, clusters_small, truncated_svd_annz_percentage, bv_annz)
    if not args.testmode:
      do_evaluations(load_sector_dataset(ds_folder), 'sector', out_folder, params_general, clusters_small, truncated_svd_annz_percentage, bv_annz)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'real_sim.scaled.bz2'), 'realsim', out_folder, params_general, clusters_medium, truncated_svd_annz_percentage, bv_annz)    
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'mediamill_static_label_scaled.bz2'), 'mediamill', out_folder, params_general, clusters_medium, truncated_svd_annz_percentage, bv_annz)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'avira_201.scaled.bz2'), 'avira201', out_folder, params_general, clusters_big, truncated_svd_annz_percentage, bv_annz)
  else:
    if not os.path.isdir(out_folder):
      raise Exception("cannot do evaluation with nonexisting output dir %s" % out_folder)

  if not os.path.isdir(output_path_latex):
    os.makedirs(output_path_latex)

  datasets_to_ignore_in_stage1 = {'caltech101': None,
                                  'kdd2001': None,
                                  'mnist800k': None,
                                  'realsim': None,
                                  'kdd': None,
                                  'e2006': None,
                                  'avira201': None,
                                  'mediamill': None}

  result_data = result_evaluation_best_params(out_folder, output_path_latex, ignore_datasets=datasets_to_ignore_in_stage1)
  best_params_pdata = remove_dataset_from_params(copy.deepcopy(result_data),
                                                 datasets_to_ignore_in_stage1)
  
  
  best_params = create_plot(output_folder=output_path_latex,
                            plot_name="kmeans_params",
                            pdata=best_params_pdata,
                            only_best_params=True)

  if not args.only_evaluation:
    if not args.testmode:
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'caltech101.scaled.bz2'), 'caltech101', out_folder, params_general, clusters_big, [best_params['pca']], [best_params['bv']])
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'kdd.scaled.bz2'), 'kdd2001', out_folder, params_general, clusters_big, [best_params['pca']], [best_params['bv']])
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'mnist800k.scaled.bz2'), 'mnist800k', out_folder, params_general, clusters_big, [best_params['pca']], [best_params['bv']])
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'e2006_static_label.scaled.bz2'), 'e2006', out_folder, params_general, clusters_small, [best_params['pca']], [best_params['bv']])
    
  result_data = result_evaluation_best_params(out_folder, output_path_latex, ignore_datasets={'caltech101': None,
                                                                                                'kdd2001': None,
                                                                                                'mnist800k': None,
                                                                                                'e2006': None})

  create_plot(output_folder=output_path_latex,
              plot_name="kmeans_params",
              pdata=copy.deepcopy(result_data),
              best_params=best_params)

  result_data = result_evaluation_best_params(out_folder, output_path_latex, remove_incomplete=True)
  
  
  create_iteration_duration_plot(output_folder=output_path_latex,
              plot_name="kmeans_params",
              pdata=copy.deepcopy(result_data),
              b=best_params['bv'],
              t=best_params['pca'])

tabs = [("speed", TABLE_TYPE_DURATION), ("memory", TABLE_TYPE_MEMORY)]
for name, table_type in tabs:
  create_table(output_folder=output_path_latex,
              plot_name="plot-%s-comparison-kmeans" % name,
              pdata=copy.deepcopy(result_data),
              best_params=best_params,
              algs_to_display={'pca_kmeans': None,
                               'bv_kmeans': None,
                               'nc_kmeans': None,
                               'kmeans': None},
              table_type=table_type)

  create_table(output_folder=output_path_latex,
              plot_name="plot-%s-comparison-elkan" % name,
              pdata=copy.deepcopy(result_data),
              best_params=best_params,
              algs_to_display={'pca_elkan': None,
                               'bv_elkan': None,
                               'elkan': None,
                               'kmeans': None},
              table_type=table_type)

  create_table(output_folder=output_path_latex,
              plot_name="plot-%s-comparison-yinyang" % name,
              pdata=copy.deepcopy(result_data),
              best_params=best_params,
              algs_to_display={'pca_yinyang': None,
                               'bv_yinyang': None,
                               'yinyang': None,
                               'kmeans': None},
              table_type=table_type)


  create_table(output_folder=output_path_latex,
              plot_name="plot-%s-comparison-all" % tabs[1][0],
              pdata=copy.deepcopy(result_data),
              best_params=best_params,
              algs_to_display={'yinyang': None,
                               'elkan': None,
                               'pca_kmeans': None,
                               'bv_kmeans': None,
                               'kmeans': None},
              table_type=tabs[1][1])