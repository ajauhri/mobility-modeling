import pickle
import matplotlib.pyplot as plt
import numpy as np
#city = "san_francisco"
city = "new_york"
prefix = "figs/ftl_" + city
"""
with open(prefix + ".pk1", 'rb') as fd:
    results = pickle.load(fd)
    stats = pickle.load(fd)
    x = []
    y1 = []
    y2 = []
    q1 = []
    q2 = []
    for k,v in results.iteritems():
        if len(v):
            v = np.array(v)
            gt = np.where(v[:,1] >= v[:,0])[0]
            lt = np.where(v[:,1] < v[:,0])[0]
            better = 100 * (v[gt, 1] - v[gt, 0]) / v[gt, 1]
            worse = 100 * (v[lt, 0] - v[lt, 1]) / v[lt, 0]
            t = len(worse) + len(better)
            if len(worse) > 0 and len(better) > 0:
                x.append(k)
                y1.append(np.mean(better))
                y2.append(np.mean(worse))
                q1.append(100*len(better)/t)
                q2.append(100*len(worse)/t)
    plt.plot(x, y2, color='r',
            label='worse average', linewidth=.2)
    plt.plot(x, y1, color='b',
            label='better average', linewidth=.2)
    plt.xlabel('time of week (3 minute slots)', fontsize=16)
    plt.ylabel('mean arrival time (seconds)', fontsize=16)
    plt.legend()
    plt.savefig(prefix + 'eta_improvements.png')
    plt.clf()
    width = .1
    plt.title("better", fontsize=16)
    plt.bar(x, q1, width=width, color='b')
    plt.xlabel("time of week (3 minute slots)", fontsize=16)
    plt.ylabel('percentage', fontsize=16)
    plt.savefig(prefix + 'better.png')
    plt.clf()
    plt.title("worse", fontsize=16)
    plt.xlabel("time of week (3 minute slots)", fontsize=16)
    plt.ylabel('percentage', fontsize=16)
    plt.ylim([0, 100])
    plt.bar(x, q2, width=width, color='b')
    plt.savefig(prefix + 'worse.png')
    plt.clf()
    fd.close()
fd_base = open("pickle/baseline_" + city + ".pk1", 'rb')
fd_ftl = open("pickle/ftl_" + city + ".pk1", 'rb')
pickle.load(fd_base)
stats_base = pickle.load(fd_base)
pickle.load(fd_ftl)
stats_ftl = pickle.load(fd_ftl)

close_pickups_base = []
poolable_pickups_base = []
close_pickups_ftl = []
poolable_pickups_ftl = []
x = []
for k,v in stats_base.iteritems():
    if k in stats_ftl:
        n_pickups = stats_base[k][2]
        assert n_pickups == stats_ftl[k][2]
        close_pickups_base.append(100*stats_base[k][0]/n_pickups)
        close_pickups_ftl.append(100*stats_ftl[k][0]/n_pickups)
        poolable_pickups_base.append(100*stats_base[k][1]/n_pickups)
        poolable_pickups_ftl.append(100*stats_ftl[k][1]/n_pickups)
        x.append(k)
print "regret:", np.mean(close_pickups_base) - np.mean(close_pickups_ftl)
width = .1
plt.title("close pickups", fontsize=16)
plt.plot(x, close_pickups_base, linewidth=width, color='b', label='Baseline')
plt.plot(x, close_pickups_ftl, linewidth=width, color='r', label='FTL')
plt.xlabel("time of week (3 minute slots)", fontsize=16)
plt.ylabel('percentage', fontsize=16)
#plt.legend()
plt.ylim([0,90])
plt.savefig('pickle/' + city + '_close_pickups.png')
plt.clf()
plt.title("poolable pickups", fontsize=16)
plt.xlabel("time of week (3 minute slots)", fontsize=16)
plt.ylabel('percentage', fontsize=16)
plt.ylim([0,16])
plt.plot(x, poolable_pickups_base, linewidth=width, color='b', label='Baseline')
plt.plot(x, poolable_pickups_ftl, linewidth=width, color='r', label='FTL')
#plt.legend()
plt.savefig('pickle/' + city+ '_poolable_pickups.png')
fd_base.close()
fd_ftl.close()
"""

def loglog_plot(city, x, y, fit_func, params, **args):
    if 'lims' in args:
        lims = args['lims']
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(args['xlabel'], fontsize=35)
    plt.ylabel(args['ylabel'], fontsize=35)
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    #plt.xlim(args['xlim'])
    #plt.ylim(args['ylim'])
    plt.plot(x, y, 'k.', markersize=6)
    plt.plot(x[lims], fit_func(x[lims], params), color='r', linewidth=2)
    #plt.title("C=%.3f, alpha=%.3f" % (params[0], params[1]), fontsize=25)
    #plt.savefig("figs/{0}_{1}.png".format(city, args['prefix']), 
    #    format='png', bbox_tight='tight')
    plt.show()
    plt.clf()

def time_series_plot(city, x, y, r, **labels):
    plt.xlabel(labels['xlabel'], fontsize=23)
    plt.ylabel(labels['ylabel'], fontsize=23)
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.plot(x, y, linewidth=1)
    plt.title("r1:{0}, r2:{1}".format(r[0], r[1]), fontsize=25)
    plt.savefig("figs/{0}_{1}.png".format(city, labels['prefix']), 
        format='png')
    plt.clf()
