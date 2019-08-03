import numpy as np
import json
from os.path import join
from collections import OrderedDict
import re
import string

general_plot_template = \
"""\\documentclass[]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usepackage{{amssymb}}
\\newcommand{{\\kmeans}}{{$k$-means}}
\\newcommand{{\\B}}{{\\mathbb{{B}}}}
\\newcommand{{\\fromRoot}}[1]{{./#1}}
\\usepackage{{mathtools}}
\\usepackage{{pgfplots}}
\\usepackage{{pgfplotstable}}
\\usepackage{{tikz}}
\\usepackage{{morefloats}}
\\usepackage{{capt-of}}
\\usepackage{{currfile}}

\\def\\pathpictureskmpp{{./pictures/}}

\\begin{{document}}
    \\begin{{centering}}
{pictures}
    \\end{{centering}}
\\end{{document}}
"""

algs_mapping = OrderedDict()
algs_mapping["kmeans++"] = "\\kmeans{}++"
algs_mapping["pca_kmeans++"] = "$\\varphi_{p}$ \\kmeans{}++"
algs_mapping["bv_kmeans++"] = "$\\varphi_{\\B}$ \\kmeans{}++"
cluster_color_mapping = {100: "blue",
                         250: "black!50",
                         500: "green",
                         1000: "brown",
                         5000: "orange",
                         10000: "purple  "}

def clearup(s, chars):
  return re.sub('[%s]' % chars, '', s).lower()

def create_variable_name(dataset, alg, no_clusters):

  dataset = clearup(dataset, string.digits)
  
  cluster_mapping = {100: "h",
                     250: "thf",
                     500: "fh",
                     1000: "m",
                     5000: "fm",
                     10000: "tm"}
  no_clusters = cluster_mapping[no_clusters]
  alg = alg.replace("_", "")
  alg = alg.replace("++", "pp")
  
  name = "%s%s%s" % (dataset, alg, no_clusters)
  return name


def create_picture_filename_tex(dataset, alg):
  name = "%s-%s.tex" % (dataset, alg)
  name = name.replace("_", "-")
  name = name.replace("++", "pp")
  return name

picture_template = \
"""
      \\begin{{figure}}
        \\centering
        \\input{{\\pathpictureskmpp {filename}}}
        \\captionof{{figure}}{{Observing the speed of {alg} for dataset {dataset} while varying the relative {parameter}}}
      \\end{{figure}}

"""
def create_picture_line(dataset, alg):
  filename = create_picture_filename_tex(dataset, alg)
  return picture_template.format(filename=filename.split(".")[0],
                                 dataset=dataset,
                                 alg=algs_mapping[alg],
                                 parameter="block vector size $b$" if "bv_" in alg else "number of eigenvectors $t$")

def create_all_pictures(pdata):
  res = []
  for dataset in pdata:
    for alg in pdata[dataset]:
      res.append(create_picture_line(dataset, alg))
  
  return "\n".join(res)

def create_table_line(dataset, alg, no_clusters):
  tmpl = "\\pgfplotstableread[col sep = semicolon]{{\\currfiledir ../data/{filename}}}\{variable_name}"
  filename = create_filename_csv(dataset, alg, no_clusters)
  variable_name = create_variable_name(dataset, alg, no_clusters)
  return tmpl.format(filename=filename,
              variable_name=variable_name), variable_name

FLOAT_FORMAT = "%.4f"

import os
import re

def make_time_percent_reference(pdata, base_alg, speedup=False):
    
    algorithms_to_delete = {base_alg:None}
    reference_times = {}
    for dataset in pdata:
      for no_clusters in pdata[dataset]['results']:
        for alg in pdata[dataset]['results'][no_clusters]:
          if alg not in [base_alg]:
            continue
          for run in range(3):
            reference_times[(dataset, no_clusters, alg, run)] = pdata[dataset]['results'][no_clusters][alg]['duration'][run][0]
          algorithms_to_delete[alg] = None
          
  
    algs_to_delete = {}
    for dataset in pdata:
      for no_clusters in pdata[dataset]['results']:
        for delalg in algorithms_to_delete:
          try:
            del pdata[dataset]['results'][no_clusters][delalg]
          except:
            pass
          
        for alg in pdata[dataset]['results'][no_clusters]:
          dat = {}
          
          for run in range(3):
            max_time = None
            #for alg_part in ["yinyang", "elkan", "kmeans"]:
            for alg_part in [base_alg]:
              #if alg_part in alg:
                if (dataset, no_clusters, alg_part, run) in reference_times:
                  max_time = reference_times[(dataset, no_clusters, alg_part, run)]
                else:
                  max_time = 100000000000.0
  
            
            if max_time > 0.0:
              
              for param_percent in pdata[dataset]['results'][no_clusters][alg]['duration'][run]:
                if param_percent not in dat:
                  dat[param_percent] = []
                if speedup:
                  percent = max_time / float(pdata[dataset]['results'][no_clusters][alg]['duration'][run][param_percent])
                else:
                  percent = float(pdata[dataset]['results'][no_clusters][alg]['duration'][run][param_percent]) / max_time
                dat[param_percent].append(percent)
                pdata[dataset]['results'][no_clusters][alg]['duration'][run][param_percent] = percent
          
          pdata[dataset]['results'][no_clusters][alg]['duration_with_error'] = {}
          for param_percent in sorted(dat.keys()):
            np_durs = np.array(dat[param_percent], dtype=np.float64)
            pdata[dataset]['results'][no_clusters][alg]['duration_with_error'][param_percent] = np.mean(np_durs), np.std(np_durs), np.min(np_durs), np.max(np_durs)
    
    for (dataset, no_clusters, alg, run) in reference_times:
      reference_times[(dataset, no_clusters, alg, run)] = reference_times[(dataset, no_clusters, base_alg, run)] / reference_times[(dataset, no_clusters, alg, run)] 
    
    return reference_times
                     
def create_object_if_not_exists(k, d, obj=OrderedDict):
  if k not in d:
    d[k] = obj()
            
def get_array_per_alg(pdata):
    pdata_new = OrderedDict()
    for dataset in pdata:
      pdata_new[dataset] = OrderedDict()
      for no_clusters in sorted(pdata[dataset]['results'].keys()):
        for alg in pdata[dataset]['results'][no_clusters]:
          create_object_if_not_exists(alg, pdata_new[dataset])
          #create_object_if_not_exists(no_clusters, pdata_new[dataset][alg], obj = list)
          
          tuples = []
          for k in pdata[dataset]['results'][no_clusters][alg]['duration_with_error']:
            speedup, stddev, min_dur, max_dur = pdata[dataset]['results'][no_clusters][alg]['duration_with_error'][k]
            tuples.append((k, speedup, stddev, min_dur, max_dur))
          tuples.sort()
          pdata_new[dataset][alg][no_clusters] = tuples
    
    return pdata_new

def create_all_csv(output_folder, pdata):
  for dataset in pdata:
    for alg in pdata[dataset]:
      for no_clusters in pdata[dataset][alg]:
        create_csv(output_folder, dataset, alg, no_clusters, pdata[dataset][alg][no_clusters])      

def create_filename_csv(dataset, alg, no_clusters):
  name = "%s-%s-%s.csv" % (dataset, alg, no_clusters)
  name = name.replace("_", "-")
  return name

def create_csv(output_folder, dataset, alg, no_clusters, results):
  name = create_filename_csv(dataset, alg, no_clusters)
  full_path = os.path.join(output_folder, name)
  with open(full_path, 'w') as f:
    for parameter, data, stddev, min_dur, max_dur in results:
      f.write("%.3f;%.3f;%.3f;%.3f;%.3f\n" % (parameter, data, stddev, min_dur, max_dur))

table_template = \
"""
{tables}

  \\begin{{tikzpicture}}  

  \\begin{{axis}}[legend cell align=left, legend entries={{{legend_entries}}}, legend pos=south east, width=10.0cm, scaled ticks=false,
  height=6cm, xlabel={x_label}, ylabel=speedup vs. \\kmeans{{}}++, grid=both, grid style={{line width=.1pt, draw=gray!10}},
  legend style={{at={{(1.01,1)}}, anchor=north west}}, minor tick num=5,
  y tick label style={{/pgf/number format/.cd,fixed,fixed zerofill,precision=2,/tikz/.cd}},
  x tick label style={{rotate=0, /pgf/number format/.cd,fixed,fixed zerofill,precision=2,/tikz/.cd}},{xlimits}]
{plots}
  \\end{{axis}}

  \\end{{tikzpicture}}
"""

"""
data is a list of tuples [('column_name1', list_of_data), ('column_name2': list_of_data), ...]
list_of_data need to be the same type! 
"""

def create_picture(output_folder, dataset, alg, pdata):  
 
  sorted_clusters = sorted(pdata[dataset][alg].keys())
  tables = []
  plots = []
  legend_entries = []
  for no_clusters in sorted_clusters:
    table_line, variable_name = create_table_line(dataset, alg, no_clusters)
    tables.append(table_line)
    plots.append(create_plot_line(cluster_color_mapping[no_clusters], variable_name))
    legend_entries.append("\emph{k=%s}" % str(no_clusters))
  
  if "bv_" in alg:
    x_label = "relative block vector size $b$"
  
  if "pca_" in alg:
    x_label = "relative no. eigenvectors $t$"
  
  pic_data = table_template.format(legend_entries=", ".join(legend_entries),
                                   tables="\n".join(tables),
                                   plots="\n".join(plots),
                                   alg=alg.replace("_", "\\_"),
                                   dataset=dataset,
                                   x_label=x_label,
                                   xlimits = "xmin=-0.01, xmax=0.425" if "pca_" in alg else "xmin=0.0, xmax=0.75")
  
  with open(join(output_folder, "pictures", create_picture_filename_tex(dataset, alg)), 'w') as f:
    f.write(pic_data)

def create_all_picture_files(output_folder, pdata):
  for dataset in pdata:
    for alg in pdata[dataset]:
      create_picture(output_folder, dataset, alg, pdata)

def create_plot_line(color, variable_name):
  tmpl = "\\addplot[mark=, color={color},line width=1.5pt] plot [error bars/.cd, y dir = both, y explicit, error bar style={{opacity=0.3}}] table[x={{0}},y={{1}}, y error index=2] {{\\{variable_name}}};"

  return tmpl.format(color=color, variable_name=variable_name)

def determine_best_params(pdata_new):
    param_speedup_bv = {}
    param_speedup_pca = {}
    for dataset in pdata_new:
      for alg in pdata_new[dataset]:
        tuples = []
        for no_clusters in pdata_new[dataset][alg]:         
          for (param, speedup, _, _, _) in pdata_new[dataset][alg][no_clusters]:
            tuples.append((speedup, param))
        tuples.sort(reverse=True)
        max_speedup, _ = tuples[0]
        if 'pca' in alg:
          param_speedup = param_speedup_pca
        if 'bv' in alg:
          param_speedup = param_speedup_bv
        for speedup, param in tuples:
          if param not in param_speedup:
            param_speedup[param] = []
          param_speedup[param].append(speedup/max_speedup)
    
    
    
    decision_tuples_bv = []
    decision_tuples_pca = []
     
    for k in param_speedup_bv:
      decision_tuples_bv.append((sum(param_speedup_bv[k]), k))

    for k in param_speedup_pca:
      decision_tuples_pca.append((sum(param_speedup_pca[k]), k))
    
    decision_tuples_bv.sort(reverse=True)
    decision_tuples_pca.sort(reverse=True)
    
    return {'bv': decision_tuples_bv[0][1],
            'pca': decision_tuples_pca[0][1]}

def create_plot(output_folder=None,
                plot_name=None,
                pdata=None,
                only_best_params=False):
     
    pname = "plot-param-search"
    py_general_filename_tex = pname + "-single.tex"
    py_sub_filename = pname
    py_sub_filename_tex = pname + ".tex"
    
    reference_times = make_time_percent_reference(pdata, 'kmeans++', speedup=True)
    
    pdata_new = get_array_per_alg(pdata)

    best_params = determine_best_params(pdata_new)
    print(best_params)
    
    if only_best_params:
      return best_params
    
    data_dir = os.path.join(output_folder, "data")
    
    if not os.path.isdir(data_dir):
      os.makedirs(data_dir)
    
    create_all_csv(data_dir, pdata_new)
    
    general_plot = general_plot_template.format(pictures=create_all_pictures(pdata_new))
        
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, py_general_filename_tex), 'wb') as f:
      f.write(general_plot)
    
    pictures_dir = os.path.join(output_folder, "pictures")
    
    if not os.path.isdir(pictures_dir):
      os.makedirs(pictures_dir)
    
    create_all_picture_files(output_folder, pdata_new)
      
    # create list of lists which contain the contents to write
    return best_params
     