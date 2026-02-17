import os
import sys

Graphs = [
        # 'LiveJournal',
        # 'Orkut',
        # 'yahoo-songs',
        # 'PubMed',
        # 'Twitter',
        # 'Friendster',
        'uk-2007'
        ]

EXES = {
    'kcore' : 'core-decomposition/core-decomposition-cpu',
    'mis' : 'independentset/maximal-independentset-cpu',
    'pr' : 'pagerank/pagerank-pull-cpu',
    'sssp' : 'sssp/sssp-cpu',
    'tc' : 'triangle-counting/triangle-counting-cpu',
}

FLAGS = {
    'kcore' : '--symmetricGraph',
    'mis' : '--symmetricGraph --algo=pull',
    'pr' : '--symmetricGraph --transposedGraph --maxIterations=100 --tolerance=0',
    'sssp' : '--symmetricGraph',
    'tc' : '--symmetricGraph --algo=edgeiterator'
}

WORKSPACE = '~/yby/Galois/build/lonestar/analytics/cpu'
DATAROOT = '/data/disk1/yangboyu/PartAndProj'
LOGDDIR = '~/yby/PartAndProj/exp'

if __name__ == '__main__':
    for graph in Graphs:
        for task, _exe in EXES.items():
            graphdir = os.path.join(DATAROOT, graph, 'galois', '{}.gr'.format(graph))
            exe = os.path.join(WORKSPACE, _exe)
            logfile = os.path.join(LOGDDIR, graph, task + '.galoislog')
            os.system('{} -t=32 {} {} > {}'.format(exe, FLAGS[task], graphdir, logfile))