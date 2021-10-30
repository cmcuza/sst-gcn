import pickle as pkl
import time
import networkx as nx
import pandas as pd
import numpy as np
from functools import reduce
from itertools import product
import operator
from config import parser
from itertools import islice
from multiprocessing import Process


ms = 0.27778
np.random.seed(0)
args = parser.parse_args()


class Counter:
    def __init__(self, total):
        self.threshold = []
        self.count = 0
        self.total = total

    def increase(self, threshold):
        self.count += 1
        self.threshold.append(threshold)

    def get_percentage(self):
        return self.count / self.total


class Diff:
    def __init__(self):
        self.diff_list = []
        self.thresholds = []

    def add_diff(self, diff, threshold):
        self.diff_list.append(diff)
        self.thresholds.append(threshold)

    def get_avg(self):
        return np.mean(self.diff_list)


class Logger:
    __instance = None
    counting_statistics = {}
    difference_statistics = {}
    map = {}
    threshold = 0
    inverse_map = {}
    threshold_map = {}
    total: int = 0
    min = 5000
    time_map: dict = {-10: min, -5: min, -2: min, 0: min, 5: min, 10: min, 15: min, 20: min}
    visited = []
    total_percentage = []
    total_lower_percentages = []
    total_lower_diffs = []
    total_higher_percentages = []
    total_higher_diffs = []
    top_late_probability = []
    k_late_probability = []

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Logger.__instance is None:
            Logger()
        return Logger.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Logger.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self

    def update_time_map(self, min):
        self.time_map: dict = {-10: min, -5: min, -2: min, 0: min, 5: min, 10: min, 15: min, 20: min}

    def add(self, OD, lenght, threshold, total, difference):
        if OD in self.inverse_map:
            if threshold in self.threshold_map[OD]:
                return

        counter = self.counting_statistics.get(OD, Counter(total))
        counter.increase(threshold)
        self.counting_statistics[OD] = counter

        diff = self.difference_statistics.get(OD, Diff())
        diff.add_diff(difference, threshold)
        self.difference_statistics[OD] = diff
        self.map.setdefault(lenght, set()).add(OD)
        self.inverse_map[OD] = lenght
        self.threshold_map.setdefault(OD, []).append(threshold)

    def compute_statistics(self, array):
        # compute count statistics
        percentages = []
        for OD, counter in self.counting_statistics.items():
            if OD in array:
                percentages.append(counter.get_percentage())

        diffs = []
        for OD, diff in self.difference_statistics.items():
            if OD in array:
                diffs.append(diff.get_avg())

        return percentages, diffs

    def report_total_statistics(self):
        print("Total small paths max percentage obtained is ", np.max(self.total_lower_percentages))
        print("Total small paths max difference obtained is ", np.max(self.total_lower_diffs))
        print("Total small paths mean percentage obtained is ", np.mean(self.total_lower_percentages))
        print("Total small paths mean difference obtained is ", np.mean(self.total_lower_diffs))

        print("Total higher paths max percentage obtained is ", np.max(self.total_higher_percentages))
        print("Total higher paths max difference obtained is ", np.max(self.total_higher_diffs))
        print("Total higher paths mean percentage obtained is ", np.mean(self.total_higher_percentages))
        print("Total higher paths mean difference obtained is ", np.mean(self.total_higher_diffs))

        print("Average probability of arriving later than threshold is", np.mean(self.top_late_probability))
        print("Max probability of arriving later than threshold is", np.max(self.top_late_probability))

        print("Average probability of arriving later than threshold is for K", np.mean(self.k_late_probability))
        print("Max probability of arriving later than threshold is for K", np.min(self.k_late_probability))

        print('Final percentage', np.mean(Logger.getInstance().total_percentage))
        round_k_late = np.round(self.k_late_probability, 3)

        print('Garantee prob', (round_k_late < 0.005).sum()/self.total)

    def report_statistics(self):
        lengths_list = np.asarray([round(k, 2) for k in self.map.keys()])
        median = round(np.median(lengths_list), 2)
        min = np.min(lengths_list)
        max = np.max(lengths_list)

        higher_distance = []
        small_distance = []

        for lenght, OD_set in self.map.items():
            if lenght <= median:
                for OD in list(OD_set):
                    small_distance.append(OD)
            else:
                for OD in list(OD_set):
                    higher_distance.append(OD)

        print(f'There are {len(small_distance)} OD at a small distance')
        print(f'There are {len(higher_distance)} OD at a large distance')

        print("Small path are from ", min, 'to', median, "meters")
        print("Large paths are from ", median, 'to', max, "meters")

        lower_percentages, lower_diffs = self.compute_statistics(small_distance)

        higher_percentages, higher_diffs = self.compute_statistics(higher_distance)

        print("Small paths max percentage obtained is ", np.max(lower_percentages))
        print("Small paths max difference obtained is ", np.max(lower_diffs))
        print("Small paths mean percentage obtained is ", np.mean(lower_percentages))
        print("Small paths mean difference obtained is ", np.mean(lower_diffs))

        print("Higher paths max percentage obtained is ", np.max(higher_percentages))
        print("Higher paths max difference obtained is ", np.max(higher_diffs))
        print("Higher paths mean percentage obtained is ", np.mean(higher_percentages))
        print("Higher paths mean difference obtained is ", np.mean(higher_diffs))

        print("Percentage", (len(small_distance)+len(higher_distance))/self.total)

        self.total_percentage.append((len(small_distance)+len(higher_distance))/self.total)
        self.total_lower_percentages += lower_percentages
        self.total_higher_percentages += higher_percentages
        self.total_lower_diffs += lower_diffs
        self.total_higher_diffs += higher_diffs

    def reset(self):
        self.counting_statistics = {}
        self.difference_statistics = {}
        self.map = {}
        self.inverse_map = {}
        self.threshold_map = {}
        self.total = 0

    def get_variable(self):
        return {'counting_statistics': self.counting_statistics,
                'difference_statistics': self.difference_statistics,
                'map': self.map,
                'inverse_map':  self.inverse_map,
                'threshold_map': self.threshold_map,
                'total': self.total,
                'visited': self.visited,
                'total_percentage': self.total_percentage,
                'total_lower_percentages': self.total_lower_percentages,
                'total_lower_diffs': self.total_lower_diffs,
                'total_higher_percentages': self.total_higher_percentages,
                'total_higher_diffs': self.total_higher_diffs,
                'late_probability': self.top_late_probability,
                'k_late_probability': self.k_late_probability
                }


def create_graph():
    with open('data/simulation/hist_data.pickle', 'rb') as f:
        hist_test = pkl.load(f)

    with open('data/simulation/avg_data.pickle', 'rb') as f:
        avg_test = pkl.load(f)

    timestamp = np.random.randint(hist_test.shape[0])
    time_hist_test = hist_test[timestamp, ...]
    time_avg_test = avg_test[timestamp, ...]

    undirected_road_network = nx.read_edgelist('data/simulation/road_network.txt')

    edge_list = pd.read_csv('data/simulation/max_dense_subgraph.txt', sep='\t', header=None,
                            names=['link', 'start_node', 'end_node'])
    edge_list['index'] = np.arange(edge_list.shape[0])

    edge_list.set_index(['link'], inplace=True)

    directed_road_network = nx.DiGraph()

    def compute_stochastic_time(dist):
        bound0 = round(dist / (args.hist[0] * ms), 2)
        bound1 = round(dist / (args.hist[0] * ms), 2)
        bound2 = round(dist / (args.hist[0] * ms), 2)
        bound3 = round(dist / (args.hist[0] * ms), 2)
        bound4 = round(dist / (args.hist[0] * ms), 2)
        return [bound0, bound1, bound2, bound3, bound4]

    def compute_expected_time(stochastic_time, hist_speed):
        expected_time = 0

        for i in range(1, len(stochastic_time)):
            expected_time += hist_speed[i-1]*(stochastic_time[i-1]+stochastic_time[i])/2

        return round(expected_time, 2)

    df = pd.read_csv("data/simulation/link.csv")
    for link in df[["Link", "Node_Start", "Node_End", 'Length']].values:
        if undirected_road_network.has_edge(str(int(link[1])), str(int(link[2]))):
            index = edge_list.loc[int(link[0])][-1]
            stochastic_time = compute_stochastic_time(link[3])
            speed = round(time_avg_test[index][0], 2)
            speed = speed if speed < 45 else 45
            #speed = compute_expected_time(stochastic_time, time_hist_test[index])
            avg_time = round(link[3] / (speed * ms), 2)
            #expected_time = compute_expected_time(stochastic_time, time_hist_test[index])

            directed_road_network.add_edge(int(link[1]), int(link[2]),
                                           mean_speed=round(speed, 2),
                                           name=int(link[0]),
                                           time=avg_time,
                                           length=round(link[3], 2),
                                           hist_speed=np.flip(time_hist_test[index]),
                                           stochastic_time=np.flip(stochastic_time))

    return directed_road_network, edge_list


def build_range(hist):
    intervals = []
    for i in range(1, len(hist)):
        interval = (hist[i - 1], hist[i])
        intervals.append(interval)

    return intervals


def create_stochastic_paths(G, path):
    discrete_list = []
    for i in range(1, len(path)):
        discrete = np.zeros(3600)
        ced = G.get_edge_data(path[i - 1], path[i])
        cst = build_range(ced['stochastic_time'])
        chs = ced['hist_speed']

        for j in range(len(cst)):
            p = chs[j]
            inter = cst[j]
            if int(inter[0]) + 1 <= int(inter[1]):
                discrete[int(inter[0]) + 1:int(inter[1])] = p * np.repeat(1. / (inter[1] - inter[0]),
                                                                          int(inter[1]) - int(inter[0]) - 1)
                discrete[int(inter[0])] += p * (int(inter[0]) + 1 - inter[0]) / (inter[1] - inter[0])
                discrete[int(inter[1])] += p * (inter[1] - int(inter[1])) / (inter[1] - inter[0])
            else:
                discrete[int(inter[0])] += p

        if abs(discrete.sum() - 1) > 0.01:
            print('Error', discrete.sum())

        discrete_list.append(discrete)

    return discrete_list


def get_random_nodes(edge_list):
    start_node = 0
    end_node = 0

    while start_node == end_node:
        start_node = edge_list['start_node'].sample(1).values[0]
        end_node = edge_list['end_node'].sample(1).values[0]

    return start_node, end_node


def cumF(a, s):
    n = len(a)
    res = 0

    for i in range(2 ** n):
        t = s
        cc = 0
        for j in range(n):
            if i & (2 ** j) != 0:
                t -= a[j]
                cc += 1
        if t >= 0:
            res += t ** n * (-1) ** cc
    den = reduce(operator.mul, [(i + 1) * a[i] for i in range(n)], 1)
    return res / den


def continuos_stochastic_cost(paths, C):
    res = 0
    for path in paths:
        prod = 1
        suma = 0
        ar = []
        for p, inter in path:
            prod *= p
            suma += inter[0]
            ar.append(inter[1] - inter[0])
        res += cumF(ar, C - suma) * prod
    return res


def get_stochastic_cost_pd(path, C):
    conv = 1
    index = int(round(C))
    try:
        for element in path:
            conv = np.convolve(element, conv)
            conv = conv[:index]
    except:
        return 0

    res = conv.sum()

    return res


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def get_avg_cost(G, path):
    avg_cost = 0
    distance = 0
    for i in range(1, len(path)):
        ced = G.get_edge_data(path[i - 1], path[i])
        avg_cost += ced['time']
        distance += ced['length']

    return avg_cost, distance


def continuos_create_stochastic_paths(G, path):
    paths = []
    st_list = []
    hs_list = []

    for i in range(1, len(path)):
        ced = G.get_edge_data(path[i - 1], path[i])
        cst = build_range(ced['stochastic_time'])
        chs = ced['hist_speed']
        st_list.append(cst)
        hs_list.append(chs)

    hist_cartesian = ([i for i in product(*hs_list)])
    interval_cartesian = ([i for i in product(*st_list)])
    for i in range(len(hist_cartesian)):
        cpath = []
        for j in range(len(path) - 1):
            cpath.append((hist_cartesian[i][j], interval_cartesian[i][j]))
        paths.append(cpath)

    return paths


def simulate_multi_paths(G, edge_list):
    start_node, end_node = get_random_nodes(edge_list)

    if (start_node, end_node) in Logger.getInstance().visited or (end_node, start_node) in Logger.getInstance().visited:
        return

    Logger.getInstance().visited.append((start_node, end_node))
    shortest_paths = k_shortest_paths(G, start_node, end_node, 5, weight='time')
    # shortest_paths = [path for path in paths if len(path) < 20]

    e_value = Logger.getInstance().threshold

    if len(shortest_paths) <= 1:
        return False

    top_avg_cost, top_dist = get_avg_cost(G, shortest_paths[0])

    if Logger.getInstance().time_map[e_value] > top_dist:
        return False

    Logger.getInstance().total += 1

    top_stochastic_path = create_stochastic_paths(G, shortest_paths[0])

    shortest_paths.pop(0)

    # thresholds = np.arange(-5, 6)*60
    thresholds = np.array([e_value])*60
    for threshold in thresholds:
        baseline_top_prob = round(get_stochastic_cost_pd(top_stochastic_path, top_avg_cost + threshold), 4)

        Logger.getInstance().top_late_probability.append(1.0-baseline_top_prob)

        best_stochastic_prob = -1.5
        for i, path in enumerate(shortest_paths):
            # print(len(path))
            # avg_cost, dist = get_avg_cost(G, path)
            # if avg_cost - top_avg_cost > 20:
            #    return

            stochastic_path = create_stochastic_paths(G, path)
            prob = round(get_stochastic_cost_pd(stochastic_path, top_avg_cost + threshold), 4)

            if baseline_top_prob < prob:
                if best_stochastic_prob < prob:
                    best_stochastic_prob = prob

        if best_stochastic_prob != -1.5:
            Logger.getInstance().add((start_node, end_node),
                                     top_dist,
                                     threshold,
                                     thresholds.shape[0],
                                     best_stochastic_prob - baseline_top_prob)

            Logger.getInstance().k_late_probability.append(1.0 - best_stochastic_prob)

    return Logger.getInstance().total == 1000


def multi_path_monte_carlo():
    G, edge_list = None, None
    for i in range(10000):
        if (i)%5 == 0:
            G, edge_list = create_graph()
            nx.write_edgelist(G, 'data/simulation/weighted_graph.txt')

        if i % 50 == 0:
            print('Doing simulation', i + 1)

        if simulate_multi_paths(G, edge_list):
            break

    Logger.getInstance().report_statistics()


def run_exp():
    exp_list = []
    for threshold in [15, 12, 10, 7, 5, 0, -5, -7, -10, -12, -15]:
        exp = Process(target=multi_path_monte_carlo, args=(threshold,))
        exp.start()
        exp_list.append(exp)
        time.sleep(2)

    for exp in exp_list:
        exp.join()


if __name__ == "__main__":
    Logger.getInstance().min = 5000
    Logger.getInstance().update_time_map(Logger.getInstance().min)
    for budget in [15]:
        Logger.getInstance().threshold = budget
        multi_path_monte_carlo()

        with open(f'data/simulation/logger_threshold_{Logger.getInstance().threshold}_min_{Logger.getInstance().min}.pickle', 'wb') as f:
            pkl.dump(Logger.getInstance().get_variable(), f)

    Logger.getInstance().report_total_statistics()