import sys
sys.path.append('/home/snikolenko/ims')
import json, argparse, cPickle

parser = argparse.ArgumentParser(description='Convert json to simlist.')
parser.add_argument('--json', dest='json', type=str, help='json filename')
parser.add_argument('--pkl', dest='pkl', type=str, help='pkl filename')
parser.add_argument('--out', dest='out', type=str, help='output filename')
parser.set_defaults(json=None, pkl=None, out='tmp.txt')
args = parser.parse_args()

if args.pkl != None:
	d = cPickle.load(open(args.pkl))
else:
	print 'No input file!'
	exit(0)


res = {}
for k in d['layers_list']:
	cur_res = { x['sf_a'] : x['mult'][0] for x in d['layers_list'][k]['sf_list'] }
	for kk,vv in cur_res.iteritems():
		if kk in res:
			print 'Already have %s' % kk
		res[kk] = max(res.get(kk, 0), vv)


with open(args.out, 'w') as ofs:
	for k,v in res.iteritems():
		ofs.write("'%s'\t'%s'\t%f\n" % (k.split('_')[0], k.split('_')[1], v))
