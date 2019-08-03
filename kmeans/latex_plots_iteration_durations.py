import numpy as np
import json
from os.path import join
from collections import OrderedDict
import copy
import pprint
import re
import string

data_suffix="durations"
data_name = "data%s" % data_suffix
picture_name = "pictures%s" % data_suffix

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
\\usepackage{{ifthen}}
\\pgfplotsset{{compat=1.9}}
\\usepackage{{capt-of}}
\\usepackage{{currfile}}

\\def\\primaryalg{{elkan}}
\\def\\secondaryalg{{yinyang}}

\\def\\pathpicturedurations{{./{picturesdir}/}}

\\begin{{document}}
    \\begin{{centering}}
{pictures}
    \\end{{centering}}
\\end{{document}}
"""

algs_mapping = OrderedDict()
algs_mapping["kmeans"] = "Lloyd \\kmeans{}"
algs_mapping["elkan"] = "Elkan"
algs_mapping["yinyang"] = "Yinyang"
algs_mapping["nc_kmeans"] = "nc \\kmeans{}"
algs_mapping["pca_kmeans"] = "$\\varphi_{p}$ \\kmeans{}"
algs_mapping["bv_kmeans"] = "$\\varphi_{\\B}$ \\kmeans{}"
algs_mapping["pca_elkan"] = "$\\varphi_{p}$ Elkan"
algs_mapping["bv_elkan"] = "$\\varphi_{\\B}$ Elkan"
algs_mapping["pca_yinyang"] = "$\\varphi_{p}$ Yinyang"
algs_mapping["bv_yinyang"] = "$\\varphi_{\\B}$ Yinyang"

algorithm_color_mapping = {'bv_elkan':'blue',
                           'bv_yinyang':'pink',
                           'bv_kmeans': 'yellow',
                           'nc_kmeans': 'magenta',
                           'elkan': 'green',
                           'kmeans': 'black',
                           'pca_elkan': 'orange',
                           'pca_kmeans': 'purple',
                           'pca_yinyang': 'red',
                           'yinyang': 'cyan'}

algorithm_position_mapping = {'kmeans': 1,
                              'elkan': 2,
                              'yinyang': 3,
                              'bv_kmeans': 4,
                              'bv_elkan': 5,
                              'bv_yinyang': 6,
                              'pca_elkan': 7,
                              'pca_kmeans': 8,
                              'pca_yinyang': 9,
                              'nc_kmeans': 10}

position_algorithm_mapping = {algorithm_position_mapping[k]:k for k in algorithm_position_mapping}

def clearup(s, chars):
  return re.sub('[%s]' % chars, '', s).lower()

def create_variable_name(dataset, no_clusters, run):
  dataset = clearup(dataset, string.digits)

  cluster_mapping = {100: "h",
                     250: "thf",
                     500: "fh",
                     1000: "m",
                     5000: "fm",
                     10000: "tm"}
  run_mapping = {0: 'zero',
                 1: 'one',
                 2: 'two'}
  no_clusters = cluster_mapping[no_clusters]
  
  name = "%s%s%s" % (dataset, no_clusters, run_mapping[run])
  return name


def create_picture_filename_tex(dataset, no_clusters, run):
  name = "%s-%s-%s.tex" % (dataset, no_clusters, run)
  name = name.replace("_", "-")
  return name

def create_picture_line(dataset, no_clusters, run):
  tmpl = "      \\input{{\\pathpicturedurations {filename}}}\\begin{{plot{variable_name}}}[]{{\\primaryalg}}{{}}{{\\secondaryalg}}{{}}\\end{{plot{variable_name}}}"
  filename = create_picture_filename_tex(dataset, no_clusters, run)
  return tmpl.format(filename=filename.split(".")[0],
                     variable_name=create_variable_name(dataset, no_clusters, run))

def create_all_pictures(reference_times):
  
  res = []
  i = 1
  for (dataset, no_clusters) in reference_times:
    for run in reference_times[(dataset, no_clusters)]:
      app_str = ""
      
      if (i % 3 == 0) and (i != 0):
        app_str = " \\clearpage"
      
      res.append(create_picture_line(dataset, no_clusters, run) + app_str)
      i += 1
  
  return "\n".join(res)

def create_table_line(data_name, dataset, no_clusters, run):
  tmpl = "\\pgfplotstableread[col sep = semicolon]{{\\thisfiledir ../{data_name}/{filename}}}\{variable_name}"
  filename = create_filename_csv(dataset, no_clusters, run)
  variable_name = create_variable_name(dataset, no_clusters, run)
  return tmpl.format(filename=filename,
              variable_name=variable_name,
              data_name=data_name), variable_name

FLOAT_FORMAT = "%.4f"

import os
import re

def get_plot_data(dat):
  algorithms = dat.keys()
  algorithms.sort()
  
  durations = []
  mem_cons = []
  stddevs = []

  for i in range(len(algorithms)):
    alg = algorithms[i]
    
    duration_values = [x[1] for x in dat[alg]['duration'].values()]
    consumption_values = [x[0] for x in dat[alg]['duration'].values()]
    
    np_durs = np.array(duration_values, dtype=np.float64)
    np_mems = np.array(consumption_values, dtype=np.float64)
    
    mem_cons.append(np.mean(np_mems))
    durations.append(np.mean(np_durs))
    stddevs.append(np.std(np_mems))

  return algorithms, mem_cons, stddevs, durations

def normalize_data(reference_times, reference_algo = 'kmeans'):
  for (dataset, no_clusters) in reference_times:
    for run in reference_times[(dataset, no_clusters)]:
      if reference_algo not in reference_times[(dataset, no_clusters)][run]:
        continue
      ref_res = copy.deepcopy(reference_times[(dataset, no_clusters)][run][reference_algo])
      
      for alg in reference_times[(dataset, no_clusters)][run]:
        current_res = reference_times[(dataset, no_clusters)][run][alg]
        if len(current_res) != len(ref_res):
          raise Exception("length of algorithm times do not match reference times")
        
        for i in range(len(ref_res)):
          current_res[i] = current_res[i] /  float(ref_res[i])

def make_time_percent_reference(pdata, b, t):
    speedup=True
    algorithms_to_delete = {"yinyang":None, "elkan":None, "kmeans":None}
    reference_times = {}
    for dataset in pdata:
      for no_clusters in pdata[dataset]['results']:
        for alg in pdata[dataset]['results'][no_clusters]:
          for run in range(3):
            if "pca" in alg:
              if t not in pdata[dataset]['results'][no_clusters][alg]['duration'][run]:
                continue
              
              v = pdata[dataset]['results'][no_clusters][alg]['iteration_durations'][run][t]
            elif "bv_" in alg:
              v = pdata[dataset]['results'][no_clusters][alg]['iteration_durations'][run][b]
            else:
              v = pdata[dataset]['results'][no_clusters][alg]['iteration_durations'][run][0]
            
            if (dataset, no_clusters) not in reference_times:
              reference_times[(dataset, no_clusters)] = {}
              
            if run not in reference_times[(dataset, no_clusters)]:
              reference_times[(dataset, no_clusters)][run] = {}
            
            reference_times[(dataset, no_clusters)][run][alg] = v
    
    return reference_times
                       
def create_all_csv(output_folder, reference_times):
  for (dataset, no_clusters) in reference_times:
    for run in reference_times[(dataset, no_clusters)]:
        create_csv(output_folder, dataset, no_clusters, run, reference_times[(dataset, no_clusters)][run])       

def create_filename_csv(dataset, no_clusters, run):
  name = "%s-%s-%s.csv" % (dataset, no_clusters, run)
  name = name.replace("_", "-")
  return name

def rem_spec(s):
  return s.replace("_", "")

def create_csv(output_folder, dataset, no_clusters, run, results):
  name = create_filename_csv(dataset, no_clusters, run)
  full_path = os.path.join(output_folder, name)
  x = sorted(list(results.keys()))
  
  ljust_dist = 0
  # create header
  with open(full_path, 'w') as f:
    header = []
    no_iterations = len(results[results.keys()[0]])
    header.append("iteration".ljust(ljust_dist))
    algs = []
    for position in sorted(position_algorithm_mapping.keys()):
      alg = position_algorithm_mapping[position]
      if alg in results:
        algs.append(alg)
        header.append(rem_spec(alg).ljust(ljust_dist))
    
    f.write(";".join(header) + "\n")
    
    for i in range(no_iterations):
      line = []
      line.append(str(i).ljust(ljust_dist))
      for alg in algs:
        line.append(("%.3f" % (results[alg][i])).ljust(ljust_dist))
      f.write(";".join(line) + "\n")
  
  """
  
    for parameter, data, stddev, min_dur, max_dur in results:
      
  """
table_template = \
"""
\\edef\\thisfiledir{{\\currfiledir}}

\\newenvironment{{plot{variable_name}}}[5][]{{%
  \\def\\mypgfmathresult{{\\pgfmathresult}}
  \\begin{{tikzpicture}}
  \\begin{{axis}}[
    ybar,
    bar shift=0pt,
    legend cell align=left,
    width=14.0cm,
    scaled ticks=false,
    height=6cm,
    xlabel=iterations,
    ylabel=iteration duration,
    grid=both,
    grid style={{line width=.1pt, draw=gray!20}},
    legend style={{at={{(0.97, 0.95)}}, anchor=north east}},
    minor tick num=4,xmin={xmin},xmax={xmax},
    yticklabel={{\\pgfmathparse{{\\tick*100}}\\pgfmathprintnumber{{\\pgfmathresult}}\\%}},
    title={{Observing iteration durations for dataset {dataset} with k={no_clusters} for run={run}}},
    #1
  ]
  
  {tables}

  \\newcommand{{\\setplotvars}}[3]{{
  
    \\ifdefined##1
    
    {plots}
    
    \\else
    \\fi
  }}
  \\setplotvars{{#2}}{{color=black, opacity=0.7}}{{#3}};
  \\setplotvars{{#4}}{{color=red, opacity=0.6}}{{#5}};
  \\let\\setplotvars\\undefined
  \\let\\{variable_name}\\undefined

}}
{{
  \\end{{axis}}
  \\end{{tikzpicture}}%
  \\let\\pgfmathresult\\mypgfmathresult
  \\let\\mypgfmathresult\\undefined
}}
"""

"""
data is a list of tuples [('column_name1', list_of_data), ('column_name2': list_of_data), ...]
list_of_data need to be the same type! 
"""

def create_picture(output_folder, data_name, picture_name, dataset, no_clusters, run, results):  
  tables = []
  plots = []
  legend_entries = []
  table_line, variable_name = create_table_line(data_name, dataset, no_clusters, run)
  tables.append(table_line)
  
  iterations = len(results[results.keys()[0]])
  for position in sorted(position_algorithm_mapping.keys()):
    alg = position_algorithm_mapping[position]
    if alg in results:
      plots.append(create_plot_line(algorithm_color_mapping[alg], alg, variable_name))
      legend_entries.append("\emph{%s}" % alg.replace("_", "\\_"))
  

  x_label = "iterations"
  
  pic_data = table_template.format(legend_entries=", ".join(legend_entries),
                                   tables="\n".join(tables),
                                   plots="\n".join(plots),
                                   alg=alg.replace("_", "\\_"),
                                   dataset=dataset,
                                   x_label=x_label,
                                   no_clusters=str(no_clusters),
                                   run=str(run),
                                   variable_name=variable_name,
                                   xmin=(float(iterations) * -0.03),
                                   xmax=((float(iterations) - 1) * 1.03))
  
  with open(join(output_folder, picture_name, create_picture_filename_tex(dataset, no_clusters, run)), 'w') as f:
    f.write(pic_data)

def create_all_picture_files(output_folder, data_name, picture_name, reference_times):
  res = []
  for (dataset, no_clusters) in reference_times:
    for run in reference_times[(dataset, no_clusters)]:
      create_picture(output_folder, data_name, picture_name, dataset, no_clusters, run, reference_times[(dataset, no_clusters)][run])
      

def create_plot_line(color, alg, variable_name):
  tmpl = """    \\ifthenelse{{\\equal{{##1}}{{{alg}}}}}{{
    \\addplot +[mark=, color={color},bar width=1, ##2, ##3] plot table[x={{iteration}},y={{{alg}}}] {{\\{variable_name}}}; \\addlegendentry{{{alg_legend}}}}};
    {{}};
  """

  return tmpl.format(color=color, variable_name=variable_name,
                     alg=alg.replace("_", ""),
                     alg_legend=algs_mapping[alg])

def create_iteration_duration_plot(output_folder=None,
                plot_name=None,
                pdata=None,
                b=None,
                t=None):
     
    pname = "tbl-iteration-duration"
    py_general_filename_tex = pname + "-single.tex"
    py_sub_filename = pname
    py_sub_filename_tex = pname + ".tex"
    
    reference_times = make_time_percent_reference(pdata, b, t)
    normalize_data(reference_times)

    data_dir = os.path.join(output_folder, data_name)
    
    if not os.path.isdir(data_dir):
      os.makedirs(data_dir)
    
    create_all_csv(data_dir, reference_times)
    
    general_plot = general_plot_template.format(pictures=create_all_pictures(reference_times), picturesdir=picture_name)
    
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, py_general_filename_tex), 'wb') as f:
      f.write(general_plot)
      
    
    pictures_dir = os.path.join(output_folder, picture_name)
    
    if not os.path.isdir(pictures_dir):
      os.makedirs(pictures_dir)
    
    create_all_picture_files(output_folder, data_name, picture_name, reference_times)