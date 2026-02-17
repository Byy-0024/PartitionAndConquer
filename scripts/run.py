import os
import sys

Graphs = [
        # 'LiveJournal',
        #   'Orkut',
        'yahoo-songs',
        'PubMed',
        #   'Twitter',
        #   'Friendster',
        #   'uk-2007'
        ]

Tasks = [
        'kcore',
         'mis',
         'pr',
         'sssp',
         'tc',
         ]

WORKSPACE = '~/yby/PartAndProj'
EXEC = os.path.join(WORKSPACE, 'bin/main')
DATAROOT = '/data/disk1/yangboyu/PartAndProj'

if __name__ == '__main__':
    for graph in Graphs:
        graphdir = os.path.join(DATAROOT, graph)
        logdir = os.path.join(WORKSPACE, 'exp', graph)
        os.system('mkdir -p {}'.format(logdir))
        for task in Tasks:
            logfile = os.path.join(logdir, '{}.pplog'.format(task))
            if task == 'pr':
                os.system('{} {} {} 1 Greedy 32 32 0.85 100 > {}'.format(EXEC, task, graphdir, logfile ))
            elif task == 'tc':
                os.system('{} {} {} merge 1 Greedy 32 32 > {}'.format(EXEC, task, graphdir, logfile ))
            else:
                os.system('{} {} {} 1 Greedy 32 32 > {}'.format(EXEC, task, graphdir, logfile ))