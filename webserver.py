#!/home/snikolenko/anaconda/bin/python
# -*- coding: utf8 -*
import matplotlib as mpl
mpl.use('Agg')

import os
from datetime import datetime,time,date,timedelta
from os import curdir,sep,path
import psycopg2,psycopg2.extras
import json
import glob
import codecs

from operator import itemgetter

import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.concurrent import Future
from tornado import gen
from tornado.ioloop import IOLoop
import tornpsql

import numpy as np
import pandas as pd

import argparse

import time
import threading
import Queue
import decimal

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import cStringIO

parser = argparse.ArgumentParser(description='IMS evaluation results webserver.')
parser.add_argument('--gifs', dest='gifs', type=str, help='directory with animated gif subdirectories')
parser.set_defaults(gifs='static/rnn_gifs')
args = parser.parse_args()

adducts = ['H', 'K', 'Na']
correct_fnames = {
	'SIM0001_twosquares_matlab'		: 'data/hmdb_sim_list.txt',
	'SIM0002_simulated_spheroid'	: 'data/true_simulated_spheroid.txt',
	'SIM0003_simple_shapes'	: 'data/true_simple_shapes.txt',
}

dataset_params = {
	'SIM0001_twosquares_matlab' : {'nrows' : 100, 'ncols' : 100},
	'SIM0002_simulated_spheroid' : {'nrows' : 100, 'ncols' : 100},
	'SIM0003_simple_shapes' : {'nrows' : 100, 'ncols' : 100},
	'mousebrain_20um' : {'nrows' : 100, 'ncols' : 50},
}

result_datasets = {
	'pipe_decoy' : 'SIM0001_twosquares_matlab',
	'pipe_spheroid' : 'SIM0002_simulated_spheroid',
	'pipe_simpleshapes' : 'SIM0003_simple_shapes',
	'pipe_mousebrain_20um' : 'mousebrain_20um'
}

correct_intensities = {}
correct_pixels = {}
correct_pixelsets = {}

eval_results = []
eval_result_names = {}

metric_fields = [ "found_mols", "tp_mols", 'auc', 'ndcg', "L1_avg_found", "L2_avg_found", "area_true_by_correct", "area_true_by_all", "area_false_by_all", "wrong_areas" ]

my_linestyles = ['-', '--', ':']

def my_print(s):
    print "[" + str(datetime.now()) + "] " + s

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return (datetime.min + obj).time().isoformat()
        else:
            return super(DateTimeEncoder, self).default(obj)

def delayed(seconds):
	def f(x):
		time.sleep(seconds)
		return x
	return f

@gen.coroutine
def async_sleep(seconds):
    yield gen.Task(IOLoop.instance().add_timeout, time.time() + seconds)

def call_in_background(f, *args):
    result = Queue.Queue(1)
    t = threading.Thread(target=lambda: result.put(f(*args)))
    t.start()
    return result

def get_id_from_slug(slug):
	return slug if slug[-1] != '/' else slug[:-1]

class AjaxHandler(tornado.web.RequestHandler):
	def run_query(self, q):
		my_print("[SQL] %s" % q)
		return self.application.db.query(q)

	def make_datatable_dict(self, draw, count, res):
		return {
			"draw":             draw,
			"recordsTotal":     count,
			"recordsFiltered":  count,
			"data":             res    
		}

	@gen.coroutine
	def get(self, query_id, slug):
		my_print("ajax %s starting..." % query_id)
		my_print("%s" % query_id)
		my_print("%s" % slug)
		draw = self.get_argument('draw', 0)
		input_id = ""
		if len(slug) > 0:
			input_id = get_id_from_slug(slug)

		if query_id in ['datasets', 'experiments']:
			orderby = sql_fields[query_id][ int(self.get_argument('order[0][column]', 0)) ]
			orderdir = self.get_argument('order[0][dir]', 0)
			limit = self.get_argument('length', 0)
			offset = self.get_argument('start', 0)
			searchval = self.get_argument('search[value]', "")
			my_print("search for : %s" % searchval)

			## queries
			q_count = sql_counts[query_id] if searchval == "" else (sql_counts[query_id + '_search'] % (searchval, searchval, searchval))
			q_res = sql_queries[query_id] if searchval == "" else (sql_queries[query_id + '_search'] % (searchval, searchval, searchval))
			if query_id in []:
				q_count = q_count % input_id
				q_res = q_res % input_id
			my_print(q_count)
			my_print(q_res + " ORDER BY %s %s LIMIT %s OFFSET %s" % (orderby, orderdir, limit, offset))
			count = int(self.run_query(q_count)[0]['count'])
			res = self.run_query(q_res + " ORDER BY %s %s LIMIT %s OFFSET %s" % (orderby, orderdir, limit, offset))
			res_dict = self.make_datatable_dict(draw, count, [[ row[x] for x in sql_fields[query_id] ] for row in res])
		elif query_id in ['metrics']:
			res_array = [ [x['dataset'], x['name'], x['num_mols_before_10percent']['H'], x['num_mols_before_10percent']['K'], x['num_mols_before_10percent']['Na'], x['name'], x['name'], x['name'], x['name']] + [ x.get(k, -1) for k in metric_fields ] for x in eval_results ]
			print '%s' % res_array
			res_dict = self.make_datatable_dict(draw, len(res_array), res_array)
		elif query_id in ['gifs']:
			res_dict = self.make_datatable_dict(draw, len(gif_results), gif_results)
		my_print("ajax %s processed, returning..." % query_id)
		# my_print("%s" % res_dict)
		self.write(json.dumps(res_dict, cls = DateTimeEncoder))

	@gen.coroutine
	def post(self, query_id):
		my_print("ajax post " + query_id)
		if query_id in ['postwordintrusion']:
			my_print("%s" % self.request.body)
			exp_id = int(self.get_argument("exp_id"))
			words = [ self.get_argument("w1"), self.get_argument("w2"), self.get_argument("w3"), self.get_argument("w4"), self.get_argument("w5"), self.get_argument("w6") ]
			chosen = int(self.get_argument("chosen"))
			my_print("Word intrusion %s chosen %s" % (words, words[chosen]))
			self.run_query("INSERT INTO exp_data VALUES (%d, '%s')" % (exp_id, json.dumps({
				"user"  : self.get_argument("user"),
				"res_id": int(self.get_argument("res_id")),
				"topic" : int(self.get_argument("topic")),
				"words" : [self.get_argument("w1"), self.get_argument("w2"), self.get_argument("w3"), self.get_argument("w4"), self.get_argument("w5"), self.get_argument("w6")],
				"ans"   : chosen,
				"true"  : int(self.get_argument("correct"))
			})) )
		elif query_id in ['upload-metric']:
			# my_print( self.request.body.decode('utf-8') )
			my_print("Adding files:")
			for jfkey in self.request.files:
				for jf in self.request.files[jfkey]:
					my_print( "\t%s" % jf["filename"] )
					with codecs.open('results_uploaded/%s' % jf["filename"], 'w', 'utf-8') as outf:
						outf.write(jf["body"])
			my_print("Processing files...")
			add_results_pipeline('results_uploaded')
			res_dict = {}
		self.write(json.dumps(res_dict, cls = DateTimeEncoder))

class IndexHandler(tornado.web.RequestHandler):
	@gen.coroutine
	def get(self):
		self.render("index.html")

html_pages = {
}

class SimpleHtmlHandlerWithId(tornado.web.RequestHandler):
	@gen.coroutine
	def get(self, id):
		my_print("Request: %s, Id: %s" % (self.request.uri, id))
		self.render( html_pages.get( self.request.uri.split('/')[1], self.request.uri.split('/')[1] + ".html") )

class SimpleHtmlHandler(tornado.web.RequestHandler):
	@gen.coroutine
	def get(self):
		my_print("Request: %s" % self.request.uri)
		self.render( html_pages.get( self.request.uri.split('/')[1], self.request.uri.split('/')[1] + ".html") )

class MyFileHandler(tornado.web.StaticFileHandler):
    def initialize(self, path):
        self.dirname, self.filename = os.path.split(path)
        super(MyFileHandler, self).initialize(self.dirname)

    def get(self, path=None, include_body=True):
        # Ignore 'path'.
        super(MyFileHandler, self).get(self.filename, include_body)


class FDRImageHandler(tornado.web.RequestHandler):
	@property
	def db(self):
		return self.application.db

	def make_fdr_image(self, res, graph_type='real', format="png"):
		'''Save image in a given format and return the StringIO object'''
		fig = plt.figure(figsize=(20,10))
		sns.set_style("darkgrid")
		if graph_type == 'est':
			ax = plt.axes()
			ax.set_ylim((0.0, 1.0))
			sns.tsplot(data=res['fdrest_pd'], time='time', unit='run', condition='add', value='fdr')
		elif graph_type == 'evst':
			ax = plt.axes()
			ax.set_ylim((0.0, 1.0))
			for i in xrange(len(adducts)):
				a = adducts[i]
				col = sns.color_palette()[i]
				gg = res['fdrest_pd'][res['fdrest_pd']['add'] == a].groupby('truefdr')
				cur_x, cur_mean, cur_std = np.array(gg['truefdr'].first()), np.array(gg['fdr'].mean()), np.array(gg['fdr'].std())
				ax.fill_between(cur_x, cur_mean - cur_std, cur_mean + cur_std, color=col, alpha=0.3)
				ax.plot(cur_x, cur_mean, label=a, color=col)
			# sns.jointplot("truefdr", "fdr", data=res['fdrest_pd'], kind='reg', xlim=(0,0.5), ylim=(0,0.5), size=10)
		elif graph_type == 'cmp':
			ks = res.keys()
			ax = plt.axes()
			ax.set_ylim((0.0, 1.0))
			for j in xrange(len(ks)):
				stl = my_linestyles[j]
				k = ks[j]
				for i in xrange(len(adducts)):
					a = adducts[i]
					my_print('\t\t%s\t%s\t%s' % (a, sns.color_palette()[i], stl) )
					ax.plot(range(1, len(res[k]["fdr_a"][a])+1), res[k]["fdr_a"][a], label=k + ', ' + a, color=sns.color_palette()[i], linestyle=stl)
		else:
			for a in adducts:
			    plt.plot(range(1, len(res["fdr_a"][a])+1), res["fdr_a"][a], label=a)
		plt.legend(loc='upper right')
		sio = cStringIO.StringIO()
		plt.savefig(sio, format=format, bbox_inches='tight')
		return sio

	@gen.coroutine
	def get(self, slug):
		my_print(slug)
		arr_slug = slug.split('/')
		graph_type = arr_slug[0]
		if len(arr_slug) == 2:
			my_print("Creating FDR image for result %s..." % arr_slug[1])
			sio = self.make_fdr_image(eval_results[eval_result_names[arr_slug[1]]], graph_type=graph_type)
		else:
			my_print("Creating FDR image for results %s..." % arr_slug[1:])
			sio = self.make_fdr_image({ k : eval_results[eval_result_names[k]] for k in arr_slug[1:]}, graph_type='cmp')
		self.set_header("Content-Type", "image/png")
		self.write(sio.getvalue())



class Application(tornado.web.Application):
	def run_query(self, q):
		my_print("[SQL] %s" % q)
		return self.db.query(q)

	def __init__(self):
		handlers = [
			(r"^/ajax/([a-z]*)", AjaxHandler),
			(r"^/ajax/([a-z]*)/(.*)", AjaxHandler),
			(r"^/ajax/(.*)", AjaxHandler),
			(r"^/fdrimage/(.*)", FDRImageHandler),
			(r"^/metrics/", SimpleHtmlHandler),
			(r"^/upload/", SimpleHtmlHandler),
			(r"^/fdr/", SimpleHtmlHandler),
			# (r"^/rnn_gifs/(.*)", tornado.web.StaticFileHandler, {'path' : '/static/rnn_gifs'}),
			(r"^/result/(.*)", SimpleHtmlHandlerWithId),
			(r"/", IndexHandler)
		]
		settings = dict(
			static_path=path.join(os.path.dirname(__file__), "static"),
			debug=True
		)
		config_db = dict(
		    host="/var/run/postgresql/",
		    db="hse",
		    user="snikolenko",
		    password=""
		)
		tornado.web.Application.__init__(self, handlers, **settings)
		# Have one global connection to the blog DB across all handlers
		# self.db = tornpsql.Connection(config_db['host'], config_db['db'], config_db['user'], config_db['password'], 5432)

def safe_mean(a):
	return 0.0 if len(a) == 0 else np.mean(a)


def compute_metrics(res):
	ds = res.get("dataset", None)
	true_w = correct_intensities.get( ds, None )
	if 'w' in res:
		sum_w = { k : np.sum(v.values()) for k,v in res['w'].iteritems() }
		sorted_w = sorted([ (k, v) for k,v in sum_w.iteritems() ], key=itemgetter(1), reverse=True)
	elif 'metric' in res:
		sorted_w = sorted([ (k, v) for k,v in res['metric'].iteritems() if v > 0 ], key=itemgetter(1), reverse=True)
	else:
		my_print('Bad result %s! Neither w nor metric.' % res['name'])
		return res

	correct = [ x[0] in true_w for x in sorted_w ] if true_w != None else None
	est_correct = [ x[0][1] in adducts for x in sorted_w ]

	def get_fdr(correct_array):
		return [ 1 - ( np.sum(correct_array[:i]) / float(i) ) for i in xrange(1, len(correct_array)) ]

	def get_est_fdr(correct_array):
		sums = [np.sum(correct_array[:i]) for i in xrange(1, len(correct_array))]
		return [ (i+1-sums[i]) / float(sums[i]) if sums[i] > 0 else 10.0 for i in xrange(len(sums)) ]

	### FDR for graphs
	num_random_runs = 20
	res["fdr"] = get_fdr(correct) if true_w != None else None
	res["fdrest"] = get_est_fdr(est_correct)
	res["fdr_a"] = {}
	res["fdrest_a"] = {}
	res["num_mols_before_10percent"] = {}
	fdr_est_dict = { 'run' : [], 'add' : [], 'fdr' : [], 'time' : [], 'truefdr' : [] }
	for a in adducts:
	    a_correct = [ x[0] in true_w for x in sorted_w if x[0][1] == a ] if true_w != None else None
	    res["fdr_a"][a] = get_fdr(a_correct) if true_w != None else None
	    res["num_mols_before_10percent"][a] = None
	    if true_w != None:
	    	tmp_wherelarger10percent = np.where(np.array(res['fdr_a'][a]) <= .1)[0]
	    	res["num_mols_before_10percent"][a] = np.max(tmp_wherelarger10percent) if len(tmp_wherelarger10percent) > 0 else 0
	    ind_est_incorrect = [ i for i in xrange(len(sorted_w)) if not (sorted_w[i][0][1] in adducts) ]
	    for run in xrange(num_random_runs):
	    	try:
	    		cur_rand_choice = np.random.choice(ind_est_incorrect, len(sorted_w)-len(ind_est_incorrect), replace=False)
	    	except:
	    		cur_rand_choice = []
	    	cur_est_correct = [ sorted_w[i][0][1] in adducts for i in xrange(len(sorted_w)) if sorted_w[i][0][1] == a or i in cur_rand_choice ]
	    	if true_w != None:
	    		fdr_xlen = min( len(cur_est_correct)-1, len(res["fdr_a"][a]) )
	    		fdr_est_dict['truefdr'].extend(res["fdr_a"][a][:fdr_xlen])
	    	else:
	    		fdr_xlen = len(cur_est_correct)-1
	    		fdr_est_dict['truefdr'].extend([0] * fdr_xlen)
	    	fdr_est_dict['add'].extend([a] * fdr_xlen )
	    	fdr_est_dict['fdr'].extend(get_est_fdr(cur_est_correct)[:fdr_xlen])
	    	fdr_est_dict['run'].extend([run] * fdr_xlen )
	    	fdr_est_dict['time'].extend( range(1, fdr_xlen+1) )
	res['fdrest_pd'] = pd.DataFrame(fdr_est_dict)

	### Metrics

	if true_w != None:
		## Basic statistics
		total_correct = np.sum(correct)
		res["found_mols"] = total_correct
		res["tp_mols"] = total_correct / float(len(correct_intensities[ds])) if true_w != None else None

		## Ranking statistics
		dcg = np.sum([np.log(2) / np.log(i + 2) for i in xrange(len(correct)) if correct[i]]) 
		perfect_dcg = np.sum([np.log(2) / np.log(i + 2) for i in xrange(total_correct)])
		res['ndcg'] = dcg / perfect_dcg if perfect_dcg > 0 else 0.0
		rank_pos = np.sum([ len(correct) - i for i in xrange(len(correct)) if correct[i] ])
		if total_correct == 0:
		 	res['auc'] = 0.0
	 	elif total_correct == len(correct):
	 		res['auc'] = 1.0
	 	else:
			res['auc'] = ((rank_pos - (total_correct * (total_correct + 1) / 2.0)) / (total_correct * (len(correct)-total_correct)))

	if 'w' in res and true_w != None:
		## Area percentages
		pixelsets = { k : set(v.keys()) for k,v in res['w'].iteritems() }
		pixelsets_correct_intersections = { k : v.intersection(correct_pixels[ds][k]) for k,v in pixelsets.iteritems() if k in correct_pixels[ds] }
		res["area_true_by_correct"] = safe_mean([ len(v) / float(len(correct_pixels[ds][k])) for k,v in pixelsets_correct_intersections.iteritems() ])
		res["area_true_by_all"] = safe_mean([ len(v) / float(len(pixelsets[k])) for k,v in pixelsets_correct_intersections.iteritems() ])
		res["area_false_by_all"] = safe_mean([ (len(pixelsets[k]) - len(v)) / float(len(pixelsets[k])) for k,v in pixelsets_correct_intersections.iteritems() ])
		res["wrong_areas"] = safe_mean([len(v) for k,v in pixelsets.iteritems() if k not in pixelsets_correct_intersections])

		## L_p metrics
		res["L2_avg_found"] = safe_mean([ np.sum([ (w.get((i,j), 0.0) - (true_w.get(m, 0.0) if (i,j) in correct_pixels[ds].get(m, {}) else 0.0) ) ** 2 for i in xrange(100) for j in xrange(100)]) for m,w in res['w'].iteritems() ])
		res["L1_avg_found"] = safe_mean([ np.sum([ np.abs(w.get((i,j), 0.0) - (true_w.get(m, 0.0) if (i,j) in correct_pixels[ds].get(m, {}) else 0.0) ) for i in xrange(100) for j in xrange(100)]) for m,w in res['w'].iteritems() ])

	return res

def add_results_pipeline(dirname):
	global eval_results
	my_print("Reading results from %s and computing metrics:" % dirname)
	for fname in glob.glob('%s/*.txt' % dirname):
		all_res = {}
		all_res['name'] = ".".join(fname.split('/')[-1].split('.')[:-1])
		if (all_res['name'] == 'decoy_dataset_chemnoise_centroids_IMS_spatial_all_adducts_full_results'):
			all_res['name'] = 'pipeline'
		if all_res['name'] in eval_result_names:
			my_print('\t...%s already processed, skipping...' % all_res['name'])
			continue

		if all_res['name'] in result_datasets:
			all_res['dataset'] = result_datasets[all_res['name']]

		with open(fname) as f:
			my_print("\t%s" % fname)
			# headers
			headers = f.readline().strip().split(',')
			all_res.update( { h : {} for h in headers[2:] } )
			# read line by line
			for line in f:
				arr = line.strip().split(',')
				key = (arr[0], arr[1])
				for i in xrange(2, len(arr)-1):
					all_res[headers[i]][key] = float(arr[i]) if arr[i] != 'nan' else 0.0
				all_res[headers[-1]][key] = 1 if arr[-1] == 'True' else 0
		if 'moc' in all_res and 'spat' in all_res and 'spec' in all_res:
			all_res['MSM'] = { k : all_res['moc'][k] * all_res['spat'].get(k, 0.0) * all_res['spec'].get(k, 0.0) for k in all_res['moc'] }
		## computing metrics
		for k in all_res:
			if k == 'name' or k == 'mz' or k =='dataset':
				continue
			cur_res = { 'name' : all_res['name'] + ', ' + k, 'dataset' : all_res.get('dataset', 'SIM0001_twosquares_matlab'), 'metric' : all_res[k] }
			my_print('\t\t%s' % cur_res['name'])
			cur_res = compute_metrics(cur_res)
			eval_results.append(cur_res)
			eval_result_names[cur_res['name']] = len(eval_results) - 1

def add_results_data(dirname):
	global eval_results
	my_print("Reading pipeline results from %s and computing metrics:" % dirname)
	for fname in glob.glob('%s/*.json' % dirname):
		with open(fname) as f:
			my_print("\t%s" % fname)
			res = json.load(f)
		if 'w' in res:
			res['w'] = { (k.split('+')[0], k.split('+')[1]) : { (int(c.split(',')[0]), int(c.split(',')[1])) : val for c,val in v.iteritems()} for k,v in res["w"].iteritems() }
		if 'name' not in res:
			res['name'] = fname.split('/')[-1][:-5]
		## computing metrics
		res = compute_metrics(res)
		eval_results.append(res)
		eval_result_names[res['name']] = len(eval_results) - 1

def read_correct_intensities():
	global correct_intensities, correct_fnames
	my_print("Reading correct results for known datasets:")
	for k, fname in correct_fnames.iteritems():
		correct_intensities[k] = {}
		correct_pixels[k] = {}
		correct_pixelsets[k] = {}
		my_print("\t%s" % fname)
		with open(fname) as f:
			for line in f:
				arr =[ x[1:-1] if x[0] == "'" else x for x in line.strip().split()]
				correct_intensities[k][(arr[0], arr[1])] = float(arr[2])
				if k == 'SIM0001_twosquares_matlab':
					correct_pixels[k][(arr[0], arr[1])] = { (x, y) : True for x in xrange(40, 88) for y in xrange(30, 81) }
				else:
					correct_pixels[k][(arr[0], arr[1])] = {}
				correct_pixelsets[k][(arr[0], arr[1])] = set(correct_pixels[k][(arr[0], arr[1])].keys())


def main():
	try:
		read_correct_intensities()
		# add_results_data('log_results')
		add_results_pipeline('results_pipeline')
		# add_results_pipeline('results_tmp')
		port = 6789
		torn_app = Application()
		http_server = tornado.httpserver.HTTPServer(torn_app)
		http_server.listen(port)
		my_print("Starting server, listening to port %d..." % port)
		## set periodic updates
		# tornado.ioloop.IOLoop.instance().add_timeout(timedelta(seconds=5), torn_app.update_all_jobs_callback)
		## start loop
		tornado.ioloop.IOLoop.instance().start()
	except KeyboardInterrupt:
		my_print( '^C received, shutting down server' )
		http_server.socket.close()


if __name__ == "__main__":
    main()



# cc = [ [ int(x.split(',')[0]), int(x.split(',')[1]) ] for x in res['w'][kk]  ]


