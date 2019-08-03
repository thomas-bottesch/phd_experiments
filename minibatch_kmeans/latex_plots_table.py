import numpy as np
import json
import pprint
from os.path import join
from collections import OrderedDict

TABLE_TYPE_DURATION = "duration"
TABLE_TYPE_MEMORY = "memory"

column_titles = {TABLE_TYPE_DURATION: "\\specialcell[c]{Speed-up relative \\\\ to Mini-Batch \\kmeans{}}",
                 TABLE_TYPE_MEMORY: "\\specialcell[c]{Memory consumption"}

general_plot_template = \
"""\\documentclass[]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usepackage{{amssymb}}
\\newcommand{{\\kmeans}}{{$k$-means}}
\\newcommand{{\\B}}{{\\mathbb{{B}}}}
\\usepackage{{mathtools}}
\\usepackage{{pgfplots}}
\\usepackage{{tikz}}
\\usepackage{{morefloats}}
\\usepackage{{colortbl}}



\\begin{{document}}
    \\begin{{centering}}
\\input{{{py_sub_filename}}}
    \\end{{centering}}
\\end{{document}}
"""

table_template = \
"""
\\newcommand{{\\specialcell}}[2][c]{{%
  \\begin{{tabular}}[#1]{{@{{}}c@{{}}}}#2\\end{{tabular}}}}
  
\\newcolumntype{{H}}{{>{{\\setbox0=\\hbox\\bgroup}}c<{{\\egroup}}@{{}}}}

\\begin{{tabular}}{{{no_columns}}}
\\toprule
\\multicolumn{{{no_info_columns}}}{{c}}{{}} & \\multicolumn{{{no_algo_columns}}}{{c}}{{{column_title}}} \\\\
\\cmidrule{{{start_algo_columns}-{end_algo_columns}}}

{header_cells} \\\\
\\midrule
{dataset_lines}
\\bottomrule
\\end{{tabular}}
"""

"""
data is a list of tuples [('column_name1', list_of_data), ('column_name2': list_of_data), ...]
list_of_data need to be the same type! 
"""

FLOAT_FORMAT = "%.4f"

import os
import re

def get_plot_data(dat):
  algorithms = dat.keys()
  algorithms.sort()
  
  durations = []
  mem_cons = []
  stddevs_duration = []
  stddevs_mem_cons = []

  for i in range(len(algorithms)):
    alg = algorithms[i]
    
    duration_values = [x[0] for x in dat[alg]['duration'].values()]
    #consumption_values = [x[0] for x in dat[alg]['mem_consumption'].values()]
    
    np_durs = np.array(duration_values, dtype=np.float64)
    np_mems = np.array(duration_values, dtype=np.float64)
    
    mem_cons.append(np.mean(np_mems))
    durations.append(np.mean(np_durs))
    stddevs_duration.append(np.std(np_durs))
    stddevs_mem_cons.append(np.std(np_mems))

  return algorithms, mem_cons, stddevs_mem_cons, durations, stddevs_duration

def complete_algorithms(pdata, algs, base_algorithm):
    datasets = pdata.keys()
    for i in range(len(datasets)):
      dataset = datasets[i]
      no_clusters_list = pdata[dataset]['results'].keys()
      no_clusters_list.sort()
      
      for j in range(len(no_clusters_list)):
        no_clusters = no_clusters_list[j]       
        algorithms, mem_cons, stddevs_mem_cons, durations, stddevs_duration = get_plot_data(pdata[dataset]['results'][no_clusters])
        for alg in algorithms:
          if alg == base_algorithm:
            continue
          if alg not in algs:
            algs[alg] = alg

def reduce_to_algorithms(pdata, algs, algs_to_display=None):

  to_remove = []
  if algs_to_display is not None:
    for alg in algs:
      if alg not in algs_to_display:
        to_remove.append(alg)

    for alg in to_remove:
      del algs[alg]

    algs_to_remove = []
    for dataset in pdata:
      for no_clusters in pdata[dataset]['results']:
        for alg in pdata[dataset]['results'][no_clusters]:
          if alg not in algs_to_display:
            algs_to_remove.append((dataset, no_clusters, alg))

    for dataset, no_clusters, alg in algs_to_remove:
      del pdata[dataset]['results'][no_clusters][alg]

def make_columns_same_length(data_list, algs, general_cells):
  column_lengths = OrderedDict()
  for column in general_cells:
    column_lengths[column] = 0
  for column in algs:
    column_lengths[column] = 0
  
  for plot_dict in data_list:
    for column in plot_dict['data_dict']:
      for string_element in plot_dict['data_dict'][column]:
        if len(string_element) > column_lengths[column]:
          column_lengths[column] = len(string_element)
  
  for plot_dict in data_list:
    for column in plot_dict['data_dict']:
      for i in range(len(plot_dict['data_dict'][column])):
        string_element = plot_dict['data_dict'][column][i]
        if len(string_element) < column_lengths[column]:
          plot_dict['data_dict'][column][i] = string_element.ljust(column_lengths[column])
  
    
def create_table_data(pdata, algs, general_cells, table_type, base_algorithm):
    data_list = []
    datasets = pdata.keys()
    datasets.sort()
    for i in range(len(datasets)):
      dataset = datasets[i]
      no_clusters_list = pdata[dataset]['results'].keys()
      no_clusters_list.sort()
      dataset_type = 'small'
      data_dict = OrderedDict()
      for k in general_cells:
        data_dict[k] = ["(mem)"] * len(no_clusters_list)
      for k in algs:
        data_dict[k] = ["(mem)"] * len(no_clusters_list)
        
      for j in range(len(no_clusters_list)):
        if j == 0:
          data_dict['dataset'][j] = ("\\multirow{%d}{*}{\\specialcell[c]{%s \\\\{\\scriptsize %d / %d / %d}}}"
                                    % (len(no_clusters_list),
                                       dataset,
                                       pdata[dataset]['infos']['input_samples'],
                                       pdata[dataset]['infos']['input_dimension'],
                                       pdata[dataset]['infos']['input_annz']))
        else:
          data_dict['dataset'][j] = ""
        no_clusters = no_clusters_list[j]
        if no_clusters > 1000 and no_clusters <= 5000:
          dataset_type = 'medium'

        if no_clusters >= 10000:
          dataset_type = 'big'
        
        data_dict['num_clusters'][j] = str(no_clusters)       
        algorithms, mem_cons, stddevs_mem_cons, durations, stddevs_duration = get_plot_data(pdata[dataset]['results'][no_clusters])

        best_algo = np.argmax(durations)
        
        if table_type == TABLE_TYPE_DURATION:
          dat = durations
          stddevs = stddevs_duration
        
        if table_type == TABLE_TYPE_MEMORY:
          dat = mem_cons
          stddevs = stddevs_mem_cons

        for m in range(len(algorithms)):
          alg = algorithms[m]
          if alg == base_algorithm:
            continue

          if table_type == TABLE_TYPE_DURATION:
            dur = "%.2f $\\pm$%.2f" % (float(dat[m]), float(stddevs[m]))
          
          if table_type == TABLE_TYPE_MEMORY:
            dur = "%.2f" % (float(dat[m]))
          
          if table_type == TABLE_TYPE_DURATION and m == best_algo:
            dur = "\\textbf{%s}" % dur
          
          try:
            data_dict[alg][j] = dur
          except:
            pprint.pprint(data_dict)
            print(alg, j, dur)
            raise
      
      plot_dict = {'data_dict': data_dict,
                   'dataset_type': dataset_type,
                   'input_dimension': pdata[dataset]['infos']['input_dimension'],
                   'input_samples': pdata[dataset]['infos']['input_samples'],
                   'input_annz': pdata[dataset]['infos']['input_annz']}
      data_list.append(plot_dict)
      
    make_columns_same_length(data_list, algs, general_cells)
    
    dataset_type_lines = OrderedDict()
    dataset_type_lines['small'] = None
    dataset_type_lines['medium'] = None
    dataset_type_lines['big'] = None
    for plot_dict in data_list:
      if dataset_type_lines[plot_dict['dataset_type']] is None:
        dataset_type_lines[plot_dict['dataset_type']] = ""
      
      if dataset_type_lines[plot_dict['dataset_type']] != "":
        dataset_type_lines[plot_dict['dataset_type']] += "\n\\arrayrulecolor{black!0}\\cmidrule{1-2}\\arrayrulecolor{black}\n"

      no_lines = len(plot_dict['data_dict']['dataset'])
      for i in range(no_lines):
        line_list = []
        for column in plot_dict['data_dict']:
          line_list.append(plot_dict['data_dict'][column][i])

        dataset_type_lines[plot_dict['dataset_type']] += " & ".join(line_list) + " \\\\\n"
    
    types_to_delete = []
    for dataset_type in dataset_type_lines:
      if dataset_type_lines[dataset_type] is None:
        types_to_delete.append(dataset_type)
    
    for dataset_type in types_to_delete:
      del dataset_type_lines[dataset_type]
    
    return dataset_type_lines

def reduce_to_best_params(pdata, best_params):
    for dataset in pdata:
      for no_clusters in pdata[dataset]['results']:
        for alg in pdata[dataset]['results'][no_clusters]:
          bp = None
          for k in best_params:
            if k in alg:
              bp = best_params[k]
          
          if bp is None:
            # the current algorithm is not optimized
            continue
          for measurment in ['duration']:
            for run in pdata[dataset]['results'][no_clusters][alg][measurment]:
              if bp not in pdata[dataset]['results'][no_clusters][alg][measurment][run]:
                res = {0: 100000000000.0}
              else:
                res = {0: pdata[dataset]['results'][no_clusters][alg][measurment][run][bp]}
              pdata[dataset]['results'][no_clusters][alg][measurment][run] = res
    
def create_table(output_folder=None,
                plot_name=None,
                pdata=None,
                best_params=None,
                algs_to_display=None,
                table_type=None):
     
    if table_type not in [TABLE_TYPE_DURATION, TABLE_TYPE_MEMORY]:
      raise Exception("Unknown table type")

    pname = plot_name
    py_general_filename_tex = pname + "-single.tex"
    py_sub_filename = pname
    py_sub_filename_tex = pname + ".tex"
    
    general_cells = OrderedDict()
    general_cells['dataset'] = "Dataset \\\\  num / dim / annz"
    general_cells['num_clusters'] = "k"
    
    general_plot = general_plot_template.format(py_sub_filename=py_sub_filename)
    base_algorithm = "minibatch_kmeans"
    algs = OrderedDict()
    algs["pca_minibatch_kmeans"] = "$\\varphi_{p}$ \\\\ Mini-Batch \\\\ \\kmeans{} \\\\ \scriptsize (t=%.2f)" % best_params['pca']
    algs["bv_minibatch_kmeans"] = "$\\varphi_{\\B}$ \\\\ Mini-Batch \\\\ \\kmeans{} \\\\ \scriptsize (b=%.2f)" % best_params['bv']
    print("Reduce to", best_params)
    reduce_to_best_params(pdata, best_params)
    complete_algorithms(pdata, algs, base_algorithm)
    reduce_to_algorithms(pdata, algs, algs_to_display=algs_to_display)
    
    general_cell_orientation = len(general_cells) * "c"
    algs_cell_orientation = len(algs) * ("r" if table_type == TABLE_TYPE_DURATION else "c")
    no_columns = general_cell_orientation + algs_cell_orientation
    
    header_cells = OrderedDict()
    for k in general_cells.keys():
      header_cells[k] = general_cells[k]
      
    for k in algs.keys():
      header_cells[k] = algs[k]

    header_cell_list = []
    for c in general_cells.keys():
      header_cell_list.append("\\specialcell[c]{%s}" % header_cells[c])
    
    for k in algs.keys():
      header_cell_list.append("\\multicolumn{1}{c}{\\specialcell[c]{%s}}" % header_cells[k])

    dataset_type_lines = create_table_data(pdata, algs, general_cells, table_type, base_algorithm)
    header_cells_str = " & ".join(header_cell_list)
    
    dataset_type_lines_keys = dataset_type_lines.keys()
    for i in range(len(dataset_type_lines_keys)):
      dataset_type = dataset_type_lines_keys[i]
      if i != len(dataset_type_lines_keys) - 1:
        dataset_type_lines[dataset_type] += "\n\\midrule\n"
    


    tbl = table_template.format(no_info_columns=len(general_cells),
                                no_algo_columns=len(algs),
                                start_algo_columns=len(general_cells) + 1,
                                end_algo_columns=len(general_cells) + len(algs),
                                header_cells=header_cells_str,
                                no_columns=no_columns,
                                dataset_lines="".join(dataset_type_lines.values()),
                                column_title=column_titles[table_type])
    
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, py_general_filename_tex), 'wb') as f:
      f.write(general_plot)
      
    with open(os.path.join(output_folder, py_sub_filename_tex), 'wb') as f:
      f.write(tbl)
      
    # create list of lists which contain the contents to write
