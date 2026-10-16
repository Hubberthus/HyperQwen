import json, sys
d = sys.argv[1]
p = '/home/hubbe/qwen-serving/models/%s/quantization_config.json' % d
c = json.load(open(p))
print('top keys:', list(c.keys()))
tc = c.get('transform_config')
if tc:
    print('tc keys:', list(tc.keys()))
    g = tc['config_groups']['rot']
    print('rot group keys:', list(g.keys()))
    a = g['apply'][0]
    print('apply[0]:', {k: v for k, v in a.items() if k != 'targets'})
    print('n targets', len(a['targets']), a['targets'][:2], a['targets'][-2:])
