import os
import sys

Graphs = [
        # 'LiveJournal',
        # 'Orkut',
        'yahoo-songs',
        'PubMed',
        # 'Twitter',
        # 'Friendster',
        # 'uk-2007'
        ]

EXES = {
    'kcore' : 'KCore',
    'mis' : 'MIS',
    'pr' : 'PageRank',
    'sssp' : 'BellmanFord',
    'tc' : 'Triangle',
}

WORKSPACE = '~/yby/ligra/apps'
DATAROOT = '/data/disk1/yangboyu/PartAndProj'
LOGDDIR = '~/yby/PartAndProj/exp'

if __name__ == '__main__':
    for graph in Graphs:
        for task, _exe in EXES.items():
            graphdir = os.path.join(DATAROOT, graph, 'ligra', 'weighted.ligra' if task == 'sssp' else 'unweighted.ligra')
            exe = os.path.join(WORKSPACE, _exe)
            logfile = os.path.join(LOGDDIR, graph, task + '.ligralog')
            os.system('{} -t 32 -s {} > {}'.format(exe, graphdir, logfile))
