import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scipy
from datetime import datetime
from time import mktime
import networkx as nx
import re
import coarsening
import time
import os


def vel_list(array_like):
    if len(array_like) > 0:
        vel_array = array_like.values[array_like.values > 0]
        if len(vel_array) <= 0:
            vel_array = []
    else:
        vel_array = []

    return vel_array


def create_theoretical_mu_std():
    base = "data/server_kdd/hist/hour_hist/hist-4/"
    for i in range(5):
        hist = base+f"hist_hour_{i}.pickle"
        with open(hist, 'rb') as f:
            hist = pkl.load(f)
            hist = hist[:].values[:].tolist()
            hist = np.asarray(hist)
            theorical_mu_std = [np.mean(hist, axis=0), np.std(hist, axis=0)]
        with open(base+f'mu_std_{i}.pkl', 'wb') as f:
            pkl.dump(theorical_mu_std, f)


class LoadKDD(object):
    def __init__(self, zero_keeping=False, hist=[0, 41, 10], time_window=15, big_threshold=38, random_sec=0.5, fold=5,
                 small_threshold=0, min_nb=4, least_threshold=0.5, receptive_field=4, coarsening_level=4):
        self.hist = hist
        self.nbin = int((self.hist[1] - self.hist[0] - 1) / self.hist[2])
        self.time_window = time_window
        self.rm_rate = random_sec
        self.zero_keeping = zero_keeping
        self.big_threshold = big_threshold
        self.small_threshold = small_threshold
        self.least_threshold = least_threshold
        self.receptive_field = receptive_field
        self.fold = fold
        self.coarsening_level = coarsening_level
        self.min_nb = min_nb
        self.base_dir = f"new_kdd/server_kdd{random_sec}/"
        self.context_dir = self.base_dir + 'context/'
        self.trips_files = self.base_dir + "csv_files/trips.csv"
        self.trip_concat_pickle = self.base_dir + "df_vel.pickle"
        self.training1 = self.base_dir + "dataSets/training/trajectories(table 5)_training.csv"
        # testing1 = self.base_dir + "dataSets/testing_phase1/trajectories(table 5)_test1.csv"
        self.training2 = self.base_dir + "dataSet_phase2/trajectories(table_5)_training2.csv"
        self.testing2 = self.base_dir + "dataSet_phase2/trajectories(table 5)_test2.csv"
        self.routes_info_file = self.base_dir + "dataSets/training/links (table 3).csv"
        self.all_dataset_pickle = self.base_dir + "all_data.pickle"
        self.historical_daytime_hist = self.base_dir + "historical_daytime_hist.pickle"
        self.train_base_dir = self.base_dir + "training/{}_sub/hist-{}/".format('zero' if zero_keeping else 'avg', self.nbin)
        self.test_base_dir = self.base_dir + "testing/{}_sub/hist-{}/".format('zero' if zero_keeping else 'avg', self.nbin)
        self.perm_file = self.base_dir + "adj_perm.pickle"
        self.graph_file = self.base_dir + "perm_graphs.pickle"
        self.edje_adj = self.base_dir + "edge_adj.pickle"
        self.all_hist_base_file = self.base_dir + "hist/all_hist/hist-{}/".format(self.nbin)
        self.hour_hist_base_file = self.base_dir + "hist/hour_hist/hist-{}/".format(self.nbin)

        np.random.seed(42)

        try:
            os.makedirs(self.context_dir)
        except OSError:
            print("Path already exists ", self.context_dir)

        try:
            os.makedirs(self.train_base_dir)
        except OSError:
            print("Path already exists ", self.train_base_dir)
        try:
            os.makedirs(self.test_base_dir)
        except OSError:
            print("Path already exists ", self.test_base_dir)
        try:
            os.makedirs(self.all_hist_base_file)
        except OSError:
            print("Path already exists ", self.all_hist_base_file)
        try:
            os.makedirs(self.hour_hist_base_file)
        except OSError:
            print("Path already exists ", self.hour_hist_base_file)

    def create_trips_from_raw_source(self):
        df1 = pd.read_csv(self.training1)
        df2 = pd.read_csv(self.training2)
        df3 = pd.read_csv(self.testing2)

        columns_names = ["link_id", "start_time", "travel_time"]

        trips1 = [trip.split("#") for travel_seq in df1["travel_seq"] for trip in travel_seq.split(";")]
        trips2 = [trip.split("#") for travel_seq in df2["travel_seq"] for trip in travel_seq.split(";")]
        trips3 = [trip.split("#") for travel_seq in df3["travel_seq"] for trip in travel_seq.split(";")]

        dataset1 = pd.DataFrame(trips1, columns=columns_names)  # 763568
        dataset2 = pd.DataFrame(trips2, columns=columns_names)  # 72491
        dataset3 = pd.DataFrame(trips3, columns=columns_names)  # 16815

        print("Shape of training 1", dataset1.shape)
        print("Shape of training 2", dataset2.shape)
        print("Shape of testing 1", dataset3.shape)

        new_dataset = pd.concat([dataset1, dataset2, dataset3])

        new_dataset['travel_time'] = new_dataset['travel_time'].astype(float)
        new_dataset['time'] = pd.to_datetime(new_dataset['start_time'])
        del new_dataset['start_time']
        dataset = new_dataset.set_index('time')

        links = pd.read_csv(self.routes_info_file)
        links["length"] = links["length"].astype(int)
        links.set_index("link_id", inplace=True)

        dataset["speed"] = (links.loc[new_dataset["link_id"].values.astype(int)]["length"].values/new_dataset["travel_time"].values)*3.6
        dataset.drop_duplicates()

        print(dataset.describe())
        print("Shape of the final dataset", new_dataset.shape)
        dataset.to_csv(self.trips_files)

    def create_concat_vel(self):
        dataset = pd.read_csv(self.trips_files)
        dataset['time'] = pd.to_datetime(dataset["time"])
        dataset.set_index("time", inplace=True)

        links_groupby = dataset.groupby(["link_id"])
        sr = "{}T".format(self.time_window)
        list_dfs = []
        for link, trips_group in links_groupby:
            linki_tt = pd.DataFrame()
            linki_resample = trips_group.speed.resample(sr).apply(vel_list)
            linki_tt[link] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')
        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_vel = df_link_tb.drop_duplicates('time')
        with open(self.trip_concat_pickle, 'wb') as f:
            pkl.dump(df_vel, f)

    def create_vel_hist(self):
        print("Reading generated df_vel.pickle file...")
        with open(self.trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel = df_vel.set_index('time')

        df_vel_hist = df_vel.apply(self.get_vel_hist_rolling, axis=0,
                                   args=([self.hist]))
        #df_vel_count = df_vel.apply(self.get_vel_count_rolling, axis=0)

        #df_vel_orig = df_vel.apply(self.get_vel_orig_rolling, axis=0)

        df_vel_hist = df_vel_hist.reset_index()
        #df_vel_count = df_vel_count.reset_index()
        #df_vel_orig = df_vel_orig.reset_index()
        edges = [i for i in range(100, 124)]
        row_notnull = pd.notnull(df_vel_hist[edges].values)

        # Get the time_index and day of week
        df_vel_hist['time_index'] = (df_vel_hist['time'].dt.hour * 60 / self.time_window +
                                     df_vel_hist['time'].dt.minute / self.time_window).astype(int)
        df_vel_hist['dayofweek'] = df_vel_hist['time'].dt.dayofweek

        #print(df_vel_result["time_index"][90:100])

        num_needed = int((df_vel_hist.shape[1]-2) * (1 - self.least_threshold))
        row_keep = row_notnull.sum(axis=1) >= num_needed

        print("The minimum number of nodes with info is", num_needed)

        df_x_all = df_vel_hist[row_keep]

        print("The number of examples for training and testing is", df_x_all.shape[0])

        #df_x_all.to_csv(self.base_dir + "csv_files/final_data.csv")

        with open(self.all_dataset_pickle, 'wb') as f:
            pkl.dump(df_x_all, f)

    def create_avg_hist_hour(self):
        print("Reading generated df_vel.pickle file...")
        with open(self.trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.time_window +
                                     df_vel['time'].dt.minute / self.time_window).astype(int)

        df_vel.set_index('time', inplace=True)

        timeday_group = df_vel.groupby(["time_index"])

        timeday_hist = []
        for timeday, group in timeday_group:
            histogram = group[:-1].apply(self.get_histogram, axis=0, args=([self.hist]))
            df = pd.DataFrame()
            for column in histogram:
                df[column] = [histogram[column].values]
            df["time_index"] = timeday
            timeday_hist.append(df)

        timeday_hist = pd.concat(timeday_hist)
        timeday_hist.set_index("time_index", inplace=True)

        with open(self.historical_daytime_hist, 'wb') as f:
            pkl.dump(timeday_hist, f)

    def create_context(self):
        with open(self.all_dataset_pickle, 'rb') as f:
            all_dataset = pkl.load(f)

        edges = all_dataset.columns[1:-2].values
        all_dataset.set_index(["time"], inplace=True)

        kf = KFold(n_splits=self.fold, shuffle=False)  # Define the split - into 5 folds
        kf.get_n_splits(all_dataset)  # returns the number of splitting iterations in the cross-validator
        i = 0
        for train_index, test_index in kf.split(all_dataset):
            X_train = all_dataset.iloc[train_index]
            X_test = all_dataset.iloc[test_index]

            max_time = X_train['time_index'].max()

            train_row_notnull = pd.notnull(X_train[edges].values)
            train_weight_array = np.zeros(train_row_notnull.shape, dtype=np.int8)
            train_weight_array[train_row_notnull] = 1

            test_row_notnull = pd.notnull(X_test[edges].values)
            test_weight_array = np.zeros(test_row_notnull.shape, dtype=np.int8)
            test_weight_array[test_row_notnull] = 1

            train_data_context = self.convert_data_context(X_train, max_time, train_weight_array, edges)
            test_data_context = self.convert_data_context(X_test, max_time, test_weight_array, edges)
            with open('data/server_kdd/context/X_training_context_{}.pickle'.format(i), 'wb') as f:
                pkl.dump(train_data_context, f)
            with open('data/server_kdd/context/X_testing_context_{}.pickle'.format(i), 'wb') as f:
                pkl.dump(test_data_context, f)
            i += 1

    def create_fold_partitions(self):
        with open(self.all_dataset_pickle, 'rb') as f:
            all_dataset = pkl.load(f)
        
        all_dataset.set_index(["time"], inplace=True)

        max_time = all_dataset['time_index'].max()

        kf = KFold(n_splits=self.fold, shuffle=False)  # Define the split - into 5 folds
        kf.get_n_splits(all_dataset)  # returns the number of splitting iterations in the cross-validator
        print(kf)
        edges = all_dataset.columns[:-2].values

        i = 0
        for train_index, test_index in kf.split(all_dataset):

            X_train = all_dataset.iloc[train_index]
            X_test = all_dataset.iloc[test_index]

            self.get_all_hist(X_train, i)

            hist_datetime = self.get_avg_hist_hour(X_train, i)

            train_row_notnull = pd.notnull(X_train[edges].values)
            train_weight_array = np.zeros(train_row_notnull.shape, dtype=np.int8)
            train_weight_array[train_row_notnull] = 1

            test_row_notnull = pd.notnull(X_test[edges].values)
            test_weight_array = np.zeros(test_row_notnull.shape, dtype=np.int8)
            test_weight_array[test_row_notnull] = 1

            t = time.time()
            Y_train = self.convert_multi_channel_array(X_train, hist_datetime, edges)
            Y_test = self.convert_multi_channel_array(X_test, hist_datetime, edges)
            print("Label for training and testing generated in:", time.time()-t, "seconds")
            print("Y_train shape ", Y_train.shape)

            t = time.time()
            X_train_multichannel, train_spatial_ctx = self.convert_random_multi_channel_array(X_train, hist_datetime, train_row_notnull, edges, self.zero_keeping)
            X_test_multichannel, test_spatial_ctx = self.convert_random_multi_channel_array(X_test, hist_datetime, test_weight_array, edges, self.zero_keeping)

            train_data_context = self.convert_data_context(X_train, max_time, train_weight_array, edges)
            test_data_context = self.convert_data_context(X_test, max_time, test_weight_array, edges)
            with open(self.context_dir + 'X_training_context_{}.pickle'.format(i), 'wb') as f:
                pkl.dump([train_data_context, train_spatial_ctx], f)
            with open(self.context_dir + 'X_testing_context_{}.pickle'.format(i), 'wb') as f:
                pkl.dump([test_data_context, test_spatial_ctx], f)

            print("Data for training and testing generated in:", time.time() - t, "seconds")

            with open(self.train_base_dir+"X_training_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_train_multichannel, f)
            with open(self.train_base_dir+"Y_training_{}.pickle".format(i), 'wb') as f:
                pkl.dump(Y_train, f)
            with open(self.test_base_dir + "X_testing_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_test_multichannel, f)
            with open(self.test_base_dir+"Y_testing_{}.pickle".format(i), 'wb') as f:
                pkl.dump(Y_test, f)
            train_weight_array = train_weight_array[self.receptive_field-1:, :]
            with open(self.train_base_dir+"weight_matrix_{}.pickle".format(i), 'wb') as f:
                pkl.dump(train_weight_array, f)
            test_weight_array = test_weight_array[self.receptive_field-1:, :]
            with open(self.test_base_dir+"weight_matrix_{}.pickle".format(i), 'wb') as f:
                pkl.dump(test_weight_array, f)

            i += 1

    def get_all_hist(self, df_hist, i):
        if not os.path.exists(self.all_hist_base_file + "hist_all_{}.pickle".format(i)):
            print("Reading generated df_vel.pickle file...")
            with open(self.trip_concat_pickle, 'rb') as f:
                df_vel = pkl.load(f)

            df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.time_window +
                                    df_vel['time'].dt.minute / self.time_window).astype(int)

            df_vel.set_index('time', inplace=True)

            df_vel = df_vel.drop(df_hist.index)
            df_vel.drop(["time_index"], axis=1, inplace=True)

            histogram = df_vel.apply(self.get_histogram, axis=0, args=([self.hist]))
            df = pd.DataFrame()
            for column in histogram:
                df[column] = [histogram[column].values]

            with open(self.all_hist_base_file + "hist_all_{}.pickle".format(i), 'wb') as f:
                pkl.dump(df, f)

    def get_avg_hist_hour(self, df_hist, i):
        if os.path.exists(self.all_hist_base_file + "hist_hour_{}.pickle".format(i)):
            print("File already exists ")
            with open(self.hour_hist_base_file + "hist_hour_{}.pickle".format(i), 'rb') as f:
                return pkl.load(f)

        print("Reading generated df_vel.pickle file...")
        with open(self.trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.time_window +
                                     df_vel['time'].dt.minute / self.time_window).astype(int)

        df_vel.set_index('time', inplace=True)

        df_vel = df_vel.drop(df_hist.index)

        timeday_group = df_vel.groupby(["time_index"])

        timeday_hist = []
        for timeday, group in timeday_group:
            histogram = group.apply(self.get_histogram, axis=0, args=([self.hist]))
            df = pd.DataFrame()
            for column in histogram:
                df[column] = [histogram[column].values]
            df["time_index"] = timeday
            timeday_hist.append(df)

        timeday_hist = pd.concat(timeday_hist)
        timeday_hist.set_index("time_index", inplace=True)

        with open(self.hour_hist_base_file + "hist_hour_{}.pickle".format(i), 'wb') as f:
            pkl.dump(timeday_hist, f)

        return timeday_hist

    def get_vel_orig_rolling(self, pdSeries_like):
        orig = []
        for i, vel_list in enumerate(pdSeries_like):
            data_list = []
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

            data_array = np.array(data_list)
            data_array = np.where(data_array >= self.big_threshold, 35, data_array)
            #data_array = data_array[data_keep]

            orig.append(data_array)

        return orig

    def get_vel_hist_rolling(self, pdSeries_like, hist_bin):
        histogram = []
        for i, vel_list in enumerate(pdSeries_like):
            data_list = []
            #print(vel_list)
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

            data_array = np.array(data_list)
            data_array = np.where(data_array >= self.big_threshold, 35, data_array)

            #data_keep = (data_array < self.big_threshold) & (
            #    data_array >= self.small_threshold)
            #data_array = data_array[data_keep]

            if len(data_array) < self.min_nb:
                histogram.append(np.nan)
                continue

            hist, bin_edges = np.histogram(data_array, range(hist_bin[0], hist_bin[1], hist_bin[2]), density=True)

            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', data_array)
                print(hist)
                histogram.append(np.nan)
                continue

            hist *= hist_bin[2]

            histogram.append(hist)

        return histogram

    def get_vel_count_rolling(self, pdSeries_like):
        count = []
        for i, vel_list in enumerate(pdSeries_like):
            data_list = []
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

            data_array = np.array(data_list)
            data_array = np.where(data_array >= self.big_threshold, 35, data_array)

            count.append(len(data_array))

        return count

    def get_histogram(self, pdSeries_like, hist_bin):
        data_list = []
        for i, vel_list in enumerate(pdSeries_like):
            #print(vel_list)
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

        data_array = np.array(data_list)
        data_array = np.where(data_array >= self.big_threshold, 35, data_array)
        #data_array = data_array[data_keep]

        if len(data_array) < self.min_nb:
            #print("No enough historical data, found just", len(data_array))
            return np.nan

        hist, bin_edges = np.histogram(data_array, range(hist_bin[0], hist_bin[1], hist_bin[2]), density=True)

        if np.isnan(hist).any():
            print('nan hist returned!')
            print('data', data_array)
            print(hist)
            return np.nan

        hist *= hist_bin[2]

        return hist

    def convert_data_context(self, df_array, max_time, missing_edges, edges):
        data_context = []
        #spatial_context = []
        N = df_array.shape[0]
        num_needed = int(len(edges) * (1 - self.rm_rate))
        for i in range(self.receptive_field-1, N):

            selected_edges = np.array([0.0] * len(edges))
            notnull_idx_i = np.where(missing_edges[i, :])[0]
            if len(notnull_idx_i) <= num_needed:
                rand_choice_idx = notnull_idx_i
            else:
                rand_choice_idx = np.random.choice(notnull_idx_i, num_needed, replace=False)

            selected_edges[rand_choice_idx] = 1.0

            current_time = df_array.iloc[i]['time_index']
            current_day = df_array.iloc[i]['dayofweek']
            object_i = []

            for r in range(self.receptive_field - 1, -1, -1):
                timeofday = np.zeros((max_time, ))
                dayofweek = np.zeros((7, ))
                timeofday[(current_time-r)%max_time] = 1.0
                previous_day_week = current_day-1 if ((current_time-i)%max_time)-current_time > 0 else current_day
                dayofweek[previous_day_week] = 1.0
                context = np.concatenate([dayofweek, timeofday])
                object_i.append(context)

            data_context.append(object_i)
            #spatial_context.append(selected_edges)

        return np.asarray(data_context)#, np.asarray(spatial_context)

    def convert_random_multi_channel_array(self, df_array, df_hist, row_notnull, edges, keep_zeros=False):
        multi_channel_array = []
        N = df_array.shape[0]
        num_needed = int(len(edges) * (1 - self.rm_rate))
        max_time = df_array['time_index'].max()
        spatial_context = []

        for i in range(self.receptive_field-1, N):
            not_selected_bool = np.array([True] * len(edges))
            selected_edges = np.array([0.0] * len(edges))
            notnull_idx_i = np.where(row_notnull[i, :])[0]
            if len(notnull_idx_i) <= num_needed:
                rand_choice_idx = notnull_idx_i
            else:
                rand_choice_idx = np.random.choice(notnull_idx_i, num_needed, replace=False)

            selected_edges[rand_choice_idx] = 1.0

            not_selected_bool[rand_choice_idx] = False
            object_i = []
            time = df_array.iloc[i]["time_index"]
            spatial_context_object = []
            time_index_z = df_array.iloc[[i-j for j in range(self.receptive_field)]]
            time_index_z = time_index_z.reset_index(drop=True).set_index('time_index')

            for r in range(self.receptive_field-1, -1, -1):
                recep_r = np.zeros((len(edges), self.nbin))
                spatial_context_r = np.ones((len(edges),))
                r_node_value = df_array.iloc[i - r]
                for j in range(len(edges)):
                    if not_selected_bool[j] and r == 0:
                        spatial_context_r[j] = 0.0
                        if not keep_zeros:
                            recep_r[j, :] = df_hist.loc[time][edges[j]][:]
                    elif r == 0:
                        recep_r[j, :] = r_node_value[edges[j]][:]
                    else:
                        if (time-r)%max_time in time_index_z.index and not pd.isnull([time_index_z.loc[(time-r)%max_time][edges[j]]]).any():
                            try:
                                recep_r[j, :] = time_index_z.loc[(time - r)%max_time][edges[j]][:]
                            except ValueError:
                                recep_r[j, :] = time_index_z.loc[(time - r)%max_time].iloc[0][edges[j]][:]
                                print(time_index_z)
                        else:
                            spatial_context_r[j] = 0.0
                            if not keep_zeros:
                                recep_r[j, :] = df_hist.loc[(time - r)%max_time][edges[j]][:]

                object_i.append(recep_r)
                spatial_context_object.append(spatial_context_r)

            multi_channel_array.append(object_i)
            spatial_context.append(spatial_context_object)

        multi_channel_array = np.array(multi_channel_array)

        return multi_channel_array, np.asarray(spatial_context)

    def convert_multi_channel_array(self, df_array, df_hist, edges):
        multi_channel_array = []
        n = df_array.shape[0]

        for i in range(self.receptive_field-1, n):
            channel_i = np.zeros((len(edges), self.nbin))
            time = df_array.iloc[i]["time_index"]
            for j in range(len(edges)):
                if pd.isnull([df_array.iloc[i][edges[j]]]).any():
                    channel_i[j, :] = df_hist.iloc[time][edges[j]][:]
                else:
                    channel_i[j, :] = df_array.iloc[i][edges[j]][:]

            multi_channel_array.append(channel_i)
        multi_channel_array = np.array(multi_channel_array)

        return multi_channel_array

    def load_adj_graph(self, do_coarsening=True):
        with open(self.edje_adj, 'rb') as f:
            edges_adj = pkl.load(f)

        if do_coarsening:
            if os.path.exists(self.perm_file) and os.path.exists(self.graph_file):
                with open(self.graph_file, 'rb') as f:
                    graphs = pkl.load(f)
            else:
                graphs, perm = coarsening.coarsen(
                    edges_adj, levels=self.coarsening_level, self_connections=False)
                with open(self.perm_file, 'wb') as f:
                    pkl.dump(perm, f)
                with open(self.graph_file, 'wb') as f:
                    pkl.dump(graphs, f)
        else:
            graphs = [edges_adj]*self.coarsening_level

        return graphs, edges_adj.shape[0]

    def coarsening_data(self):
        if not os.path.exists(self.perm_file):
            self.load_adj_graph()

        if not os.path.exists(self.train_base_dir + "X_training_0.pickle"):
            self.create_fold_partitions()

        with open(self.perm_file, 'rb') as f:
            perm = pkl.load(f)

        for i in range(self.fold):
            with open(self.train_base_dir + "X_training_{}.pickle".format(i), 'rb') as f:
                X_train = pkl.load(f)

            with open(self.test_base_dir + "X_testing_{}.pickle".format(i), 'rb') as f:
                X_test = pkl.load(f)

            X_train_coarsed = coarsening.perm_data_hist_hist(X_train, perm)
            X_test_coarsed = coarsening.perm_data_hist_hist(X_test, perm)

            with open(self.train_base_dir + "X_training_coarsed_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_train_coarsed, f)
            with open(self.test_base_dir + "X_testing_coarsed_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_test_coarsed, f)

    def switch_timestamps(self):
        for i in range(5):
            with open(self.train_base_dir + "X_training_{}.pickle".format(i), 'rb') as f:
                x_train = pkl.load(f)
            with open(self.test_base_dir + "X_testing_{}.pickle".format(i), 'rb') as f:
                x_test = pkl.load(f)
            x_train = x_train[:, [3,2,1,0], :, :]
            x_test = x_test[:, [3, 2, 1, 0], :, :]
            with open(self.test_base_dir + "X_testing_switched_{}.pickle".format(i), 'wb') as f:
                pkl.dump(x_test, f)
            with open(self.train_base_dir + "X_training_switched_{}.pickle".format(i), 'wb') as f:
                pkl.dump(x_train, f)


class LoadChengdu(object):
    def __init__(self, zero_keeping=False, hist=[0, 41, 10], time_window=20, big_threshold=40, random_sec=0.5, fold=5,
                 small_threshold=1, min_nb=4, least_threshold=0.5, receptive_field=4, coarsening_level=4, num_nodes=40):
        self.__time_window = time_window
        self.__hist = hist
        self.__nbin = int((self.__hist[1] - self.__hist[0] - 1) / self.__hist[2])
        self.__rm_rate = random_sec
        self.__zero_keeping = zero_keeping
        self.__big_threshold = big_threshold
        self.__small_threshold = small_threshold
        self.__least_threshold = least_threshold
        self.__receptive_field = receptive_field
        self.__fold = fold
        self.__coarsening_level = coarsening_level
        self.__min_nb = min_nb
        self.__base_dir = f"data_zero/chengdu{num_nodes}/chengdu{random_sec}/"
        self.__context = self.__base_dir + 'context/'
        self.__full_traffic_data = self.__base_dir + "full_traffic_data/"
        self.__filter_traffic_data = self.__base_dir + "filter_traffic_data/"
        self.__subgraph = self.__base_dir + "max_dense_subgraph.txt"
        self.__edge_dict = self.__base_dir + "edge_dict.pkl"
        self.__trips_files = self.__base_dir + "trips.csv"
        self.__trip_concat_pickle = self.__base_dir + "df_vel.pickle"
        self.__all_dataset_pickle = self.__base_dir + "all_data.pickle"
        self.__historical_daytime_hist = self.__base_dir + "historical_daytime_hist.pickle"
        self.__all_hist_base_file = self.__base_dir + "hist/all_hist/hist-{}/".format(self.__nbin)
        self.__hour_hist_base_file = self.__base_dir + "hist/hour_hist/hist-{}/".format(self.__nbin)
        self.__train_base_dir = self.__base_dir + "training/{}_sub/hist-{}/".format('zero' if zero_keeping else 'avg', self.__nbin)
        self.__test_base_dir = self.__base_dir + "testing/{}_sub/hist-{}/".format('zero' if zero_keeping else 'avg', self.__nbin)
        self.__edge_adj_file = self.__base_dir + "edge_adj.pickle"
        self.__edges_file = self.__base_dir + "edges.pickle"
        self.__perm_file = self.__base_dir + "adj_perm.pickle"
        self.__graph_file = self.__base_dir + "perm_graphs.pickle"
        np.random.seed(42)

        try:
            os.makedirs(self.__filter_traffic_data)
        except OSError:
            print("Path already exists ", self.__filter_traffic_data)
        try:
            os.makedirs(self.__context)
        except OSError:
            print("Path already exists ", self.__context)

        try:
            os.makedirs(self.__train_base_dir)
        except OSError:
            print("Path already exists ", self.__train_base_dir)
        try:
            os.makedirs(self.__test_base_dir)
        except OSError:
            print("Path already exists ", self.__test_base_dir)
        try:
            os.makedirs(self.__all_hist_base_file)
        except OSError:
            print("Path already exists ", self.__all_hist_base_file)
        try:
            os.makedirs(self.__hour_hist_base_file)
        except OSError:
            print("Path already exists ", self.__hour_hist_base_file)

    def filter_data(self):
        dense_subgraph = pd.read_csv(self.__subgraph, sep='\t', header=None, names=["Link", "Node_Start", "Node_End"])

        for (dirpath, dirnames, filenames) in os.walk(self.__full_traffic_data):
            for filename in filenames:
                speed_data = pd.read_csv(os.sep.join([dirpath, filename]))
                filter_speed = speed_data[speed_data["Link"].isin(dense_subgraph["Link"])]
                filter_speed.to_csv(self.__filter_traffic_data+filename, index=None)

    def __create_timestamp(self, t, filename):
        month_day = int(re.findall("\d\d\d", filename)[0])
        tm_year = 2015
        tm_mday = month_day%100
        tm_mon = int((month_day - (month_day%100))/100)
        real_time = "{}-{}-{} " + t
        real_time = real_time.format(tm_year, tm_mon, tm_mday)
        timestamp = time.strptime(real_time, "%Y-%m-%d %H:%M")
        dt = datetime.fromtimestamp(mktime(timestamp))
        return dt

    def build_trajectories_data(self):
        trajectories = []
        for (dirpath, dirnames, filenames) in os.walk(self.__filter_traffic_data):
            for filename in filenames:
                speed_data = pd.read_csv(os.sep.join([dirpath, filename]))
                speed_data["Time"] = [self.__create_timestamp(p.split('-')[0], filename) for p in speed_data["Period"]]
                speed_data.drop(["Period"], axis=1, inplace=True)
                trajectories.append(speed_data.set_index(["Time"]))

        trajectories = pd.concat(trajectories)
        trajectories.to_csv(self.__trips_files)

    def create_concat_vel(self):
        dataset = pd.read_csv(self.__trips_files)
        dataset['time'] = pd.to_datetime(dataset["Time"])
        dataset.set_index("time", inplace=True)

        links_groupby = dataset.groupby(["Link"])
        sr = "{}T".format(self.__time_window)
        list_dfs = []
        for link, trips_group in links_groupby:
            linki_tt = pd.DataFrame()
            linki_resample = trips_group.Speed.resample(sr).apply(vel_list)
            linki_tt[link] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')
        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_vel = df_link_tb.drop_duplicates('time')

        with open(self.__trip_concat_pickle, 'wb') as f:
            pkl.dump(df_vel, f)

    def create_vel_hist(self):
        print("Reading generated df_vel.pickle file...")
        with open(self.__trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel = df_vel.set_index('time')

        df_vel_hist = df_vel.apply(self.__get_vel_hist_rolling, axis=0,
                                   args=([self.__hist]))

        df_vel_hist = df_vel_hist.reset_index()
        row_notnull = pd.notnull(df_vel_hist[df_vel_hist.columns].values)

        # Get the time_index and day of week
        df_vel_hist['time_index'] = (df_vel_hist['time'].dt.hour * 60 / self.__time_window +
                                     df_vel_hist['time'].dt.minute / self.__time_window).astype(int)
        df_vel_hist['dayofweek'] = df_vel_hist['time'].dt.dayofweek

        num_needed = int((df_vel_hist.shape[1]-2) * (1 - self.__least_threshold))
        row_keep = row_notnull.sum(axis=1) >= num_needed

        print("The minimum number of nodes with info is", num_needed)

        df_x_all = df_vel_hist[row_keep]

        print("The number of examples for training and testing is", df_x_all.shape[0])

        with open(self.__all_dataset_pickle, 'wb') as f:
            pkl.dump(df_x_all, f)

    def create_avg_hist_hour(self):
        print("Reading generated df_vel.pickle file...")
        with open(self.__trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.__time_window +
                                     df_vel['time'].dt.minute / self.__time_window).astype(int)
        df_vel.set_index('time', inplace=True)

        timeday_group = df_vel.groupby(["time_index"])

        timeday_hist = []
        for timeday, group in timeday_group:
            histogram = group[:-1].apply(self.__get_histogram, axis=0, args=([self.__hist]))
            df = pd.DataFrame()
            for column in histogram:
                if isinstance(column, int):
                    df[column] = [histogram[column].values]
            df["time_index"] = timeday
            timeday_hist.append(df)

        timeday_hist = pd.concat(timeday_hist)
        timeday_hist.set_index("time_index", inplace=True)

        with open(self.__historical_daytime_hist, 'wb') as f:
            pkl.dump(timeday_hist, f)

    def create_fold_partitions(self):
        with open(self.__all_dataset_pickle, 'rb') as f:
            all_dataset = pkl.load(f)

        all_dataset["time"] = pd.to_datetime(all_dataset["time"])

        all_dataset.set_index(["time"], inplace=True)

        edges = all_dataset.columns[:-2].values
        test_intervals = [(pd.datetime(year=2015, month=6, day=1), pd.datetime(year=2015, month=6, day=10)),
                          (pd.datetime(year=2015, month=6, day=10), pd.datetime(year=2015, month=6, day=19)),
                          (pd.datetime(year=2015, month=6, day=19), pd.datetime(year=2015, month=6, day=28)),
                          (pd.datetime(year=2015, month=6, day=28), pd.Timestamp(year=2015, month=7, day=7)),
                          (pd.datetime(year=2015, month=6, day=7), pd.datetime(year=2015, month=6, day=16))]
        i = 0

        time_index = set(all_dataset.time_index.values)
        receptive_field = {}
        for element in time_index:
            if element-1 in receptive_field:
                receptive_field[element] = receptive_field[element-1] + 1
            else:
                receptive_field[element] = 1

        for test_index_init, test_index_end in test_intervals:
            mask = ((all_dataset.index >= test_index_init) & (all_dataset.index < test_index_end))
            X_train = all_dataset.loc[np.logical_not(mask)]
            X_test = all_dataset.loc[mask]

            self.__get_all_hist(X_test, i)
            hist_datetime = self.__get_avg_hist_hour(X_test, i)
            print("The number of training example is", X_train.shape[0], "the number of testing", X_test.shape[0])

            train_row_notnull = pd.notnull(X_train[edges].values)
            train_weight_array = np.zeros(train_row_notnull.shape, dtype=np.int8)
            train_weight_array[train_row_notnull] = 1

            test_row_notnull = pd.notnull(X_test[edges].values)
            test_weight_array = np.zeros(test_row_notnull.shape, dtype=np.int8)
            test_weight_array[test_row_notnull] = 1

            t = time.time()

            X_train_multichannel, train_spatial_ctx = self.__convert_random_multi_channel_array_prob(X_train, hist_datetime, train_row_notnull,
                                                                             edges, receptive_field, self.__zero_keeping)
            with open(self.__train_base_dir + "X_training_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_train_multichannel, f)

            X_test_multichannel, test_spatial_ctx = self.__convert_random_multi_channel_array_prob(X_test, hist_datetime, test_weight_array,
                                                                            edges, receptive_field, self.__zero_keeping)
            with open(self.__test_base_dir + "X_testing_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_test_multichannel, f)

            train_data_context = self.convert_data_context(X_train, receptive_field)
            test_data_context = self.convert_data_context(X_test, receptive_field)

            with open(self.__context + "X_training_context_{}.pickle".format(i), 'wb') as f:
                pkl.dump([train_data_context, train_spatial_ctx], f)
            with open(self.__context + "X_testing_context_{}.pickle".format(i), 'wb') as f:
                pkl.dump([test_data_context, test_spatial_ctx], f)

            print("Data for training and testing generated in:", time.time() - t, "seconds")

            t = time.time()

            Y_train = self.__convert_multi_channel_array(X_train, hist_datetime, receptive_field, edges)
            Y_test = self.__convert_multi_channel_array(X_test, hist_datetime, receptive_field, edges)
            print("Label for training and testing generated in:", time.time() - t, "seconds")
            print("Y_train shape ", Y_train.shape)

            with open(self.__train_base_dir + "Y_training_{}.pickle".format(i), 'wb') as f:
                pkl.dump(Y_train, f)
            with open(self.__test_base_dir + "Y_testing_{}.pickle".format(i), 'wb') as f:
                pkl.dump(Y_test, f)
            with open(self.__train_base_dir + "weight_matrix_{}.pickle".format(i), 'wb') as f:
                pkl.dump(train_weight_array, f)
            with open(self.__test_base_dir + "weight_matrix_{}.pickle".format(i), 'wb') as f:
                pkl.dump(test_weight_array, f)

            i += 1

    def __get_avg_hist_hour(self, df_hist, i):
        if os.path.exists(self.__hour_hist_base_file + "hist_hour_{}.pickle".format(i)):
            print("File already exists ")
            with open(self.__hour_hist_base_file + "hist_hour_{}.pickle".format(i), 'rb') as f:
                return pkl.load(f)

        print("Reading generated df_vel.pickle file...")
        with open(self.__trip_concat_pickle, 'rb') as f:
            df_vel = pkl.load(f)

        df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.__time_window +
                                     df_vel['time'].dt.minute / self.__time_window).astype(int)

        df_vel.set_index('time', inplace=True)

        df_vel = df_vel.drop(df_hist.index)

        timeday_group = df_vel.groupby(["time_index"])

        timeday_hist = []
        for timeday, group in timeday_group:
            histogram = group.apply(self.__get_histogram, axis=0, args=([self.__hist]))
            df = pd.DataFrame()
            for column in histogram:
                if isinstance(column, int):
                    df[column] = [histogram[column].values]
            df["time_index"] = timeday
            timeday_hist.append(df)

        timeday_hist = pd.concat(timeday_hist)
        timeday_hist.set_index("time_index", inplace=True)

        with open(self.__hour_hist_base_file + "hist_hour_{}.pickle".format(i), 'wb') as f:
            pkl.dump(timeday_hist, f)

        return timeday_hist

    def __get_vel_hist_rolling(self, pdSeries_like, hist_bin):
        histogram = []
        for i, vel_list in enumerate(pdSeries_like):
            data_list = []
            #print(vel_list)
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

            data_array = np.array(data_list)
            data_array = np.where(data_array >= self.__big_threshold, 35, data_array)
            #data_keep = (data_array < self.__big_threshold) & (data_array >= self.__small_threshold)
            #data_array = data_array[data_keep]

            if len(data_array) < self.__min_nb:
                histogram.append(np.nan)
                continue

            hist, bin_edges = np.histogram(data_array, range(hist_bin[0], hist_bin[1], hist_bin[2]), density=True)

            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', data_array)
                print(hist)
                histogram.append(np.nan)
                continue

            hist *= hist_bin[2]

            histogram.append(hist)

        return histogram

    def __get_histogram(self, pdSeries_like, hist_bin):
        data_list = []
        for i, vel_list in enumerate(pdSeries_like):
            #print(vel_list)
            if type(vel_list) == list or type(vel_list) == np.ndarray:
                for item in vel_list:
                    data_list.append(item)
            else:
                data_list.append(vel_list)

        data_array = np.array(data_list)
        data_array = np.where(data_array >= self.__big_threshold, 35, data_array)

        if len(data_array) < self.__min_nb:
            #print("No enough historical data, found just", len(data_array))
            return np.nan

        hist, bin_edges = np.histogram(data_array, range(hist_bin[0], hist_bin[1], hist_bin[2]), density=True)

        if np.isnan(hist).any():
            print('nan hist returned!')
            print('data', data_array)
            print(hist)
            return np.nan

        hist *= hist_bin[2]

        return hist

    def __get_all_hist(self, df_hist, i):
        if not os.path.exists(self.__all_hist_base_file + "hist_all_{}.pickle".format(i)):
            print("Reading generated df_vel.pickle file...")
            with open(self.__trip_concat_pickle, 'rb') as f:
                df_vel = pkl.load(f)

            df_vel['time_index'] = (df_vel['time'].dt.hour * 60 / self.__time_window +
                                    df_vel['time'].dt.minute / self.__time_window).astype(int)

            df_vel.set_index('time', inplace=True)

            df_vel = df_vel.drop(df_hist.index)
            df_vel.drop(["time_index"], axis=1, inplace=True)

            histogram = df_vel.apply(self.__get_histogram, axis=0, args=([self.__hist]))
            df = pd.DataFrame()
            for column in histogram:
                if isinstance(column, int):
                    df[column] = [histogram[column].values]

            with open(self.__all_hist_base_file + "hist_all_{}.pickle".format(i), 'wb') as f:
                pkl.dump(df, f)

    def __convert_multi_channel_array(self, df_array, df_hist, receptive_field, edges):
        multi_channel_array = []
        all_shape = df_array.shape

        for i in range(all_shape[0]):
            channel_i = np.zeros((len(edges), self.__nbin))
            time = df_array.iloc[i]["time_index"]

            if receptive_field[time] < self.__receptive_field:
                continue

            for j in range(len(edges)):
                if pd.isnull([df_array.iloc[i][edges[j]]]).any():
                    for k in range(self.__nbin):
                        channel_i[j, k] = df_hist.iloc[time][edges[j]][k]
                else:
                    for k in range(self.__nbin):
                        channel_i[j, k] = df_array.iloc[i][edges[j]][k]

            multi_channel_array.append(channel_i)
        multi_channel_array = np.array(multi_channel_array)

        return multi_channel_array

    def __convert_random_multi_channel_array(self, df_array, df_hist, row_notnull, edges, receptive_field, keep_zeros=False):
        multi_channel_array = []
        all_shape = df_array.shape
        num_needed = int(len(edges) * (1 - self.__rm_rate))
        for i in range(all_shape[0]):
            not_selected_bool = np.array([True] * len(edges))
            notnull_idx_i = np.where(row_notnull[i, :])[0]
            rand_choice_idx = np.random.choice(notnull_idx_i, num_needed, replace=False)
            not_selected_bool[rand_choice_idx] = False
            object_i = []
            time = df_array.iloc[i]["time_index"]
            if receptive_field[time] < self.__receptive_field:
                continue
            for r in range(self.__receptive_field):
                recep_r = np.zeros((len(edges), self.__nbin))
                for j in range(len(edges)):
                    if not_selected_bool[j]:
                        if not keep_zeros:
                            for k in range(self.__nbin):
                                recep_r[j, k] = df_hist.loc[(time - r)][edges[j]][k]
                    else:
                        if r == 0:
                            for k in range(self.__nbin):
                                recep_r[j, k] = df_array.iloc[i][edges[j]][k]
                        else:
                            if i-r < 0 or pd.isnull([df_array.iloc[i-r][edges[j]]]).any() or df_array.iloc[i-r].time_index != (time-r):
                                for k in range(self.__nbin):
                                    recep_r[j, k] = df_hist.loc[(time-r)][edges[j]][k]
                            else:
                                for k in range(self.__nbin):
                                    recep_r[j, k] = df_array.iloc[i-r][edges[j]][k]
                #print(recep_r)
                object_i.append(recep_r)
            multi_channel_array.append(object_i)

        multi_channel_array = np.array(multi_channel_array)

        return multi_channel_array

    def __convert_random_multi_channel_array_prob(self, df_array, df_hist, row_notnull, edges, receptive_field,
                                                  keep_zeros=False):
        multi_channel_array = []
        all_shape = df_array.shape
        num_needed = int(len(edges) * (1 - self.__rm_rate))
        np.random.seed(42)
        np.random.RandomState(42)
        sigma = 0.15 * len(edges)
        mu = 0.3 * len(edges)
        min = 0.5 * len(edges)
        spatial_context = []
        for i in range(all_shape[0]):
            not_selected_bool = np.array([True] * len(edges))
            selected_edges = np.array([0.0] * len(edges))

            notnull_idx_i = np.where(row_notnull[i, :])[0]
            rand_choice_idx = np.random.choice(notnull_idx_i, num_needed, replace=False)

            not_selected_bool[rand_choice_idx] = False
            selected_edges[rand_choice_idx] = 1.0

            object_i = []
            time = df_array.iloc[i]["time_index"]

            if receptive_field[time] < self.__receptive_field:
                continue

            for r in range(self.__receptive_field):
                recep_r = np.zeros((len(edges), self.__nbin))
                selected_bool_r = np.array([False] * len(edges))
                num_needed_r = int(sigma * np.random.randn() + mu)
                num_needed_r = int(mu) if num_needed_r > min or num_needed_r < 0 else num_needed_r
                rand_choice_idx_r = np.random.choice(len(edges), num_needed_r, replace=False)
                selected_bool_r[rand_choice_idx_r] = True

                for j in range(len(edges)):
                    if not_selected_bool[j] and r == 0:
                        if not keep_zeros:
                            for k in range(self.__nbin):
                                recep_r[j, k] = df_hist.loc[(time - r)][edges[j]][k]
                    else:
                        if r == 0:
                            for k in range(self.__nbin):
                                recep_r[j, k] = df_array.iloc[i][edges[j]][k]
                        else:
                            if i - r < 0 or selected_bool_r[j] or pd.isnull([df_array.iloc[i - r][edges[j]]]).any() or \
                                    df_array.iloc[i - r].time_index != (time - r):
                                if not keep_zeros:
                                    for k in range(self.__nbin):
                                        recep_r[j, k] = df_hist.loc[(time - r)][edges[j]][k]
                            else:
                                for k in range(self.__nbin):
                                    recep_r[j, k] = df_array.iloc[i - r][edges[j]][k]

                object_i.append(recep_r)

            multi_channel_array.append(object_i)
            spatial_context.append(selected_edges)

        multi_channel_array = np.array(multi_channel_array)
        spatial_context = np.array(spatial_context)

        return multi_channel_array, spatial_context

    def build_edge_adjacency(self):
        with open(self.__edge_dict, 'rb') as f:
            edge_dict = pkl.load(f)

        graph = nx.from_dict_of_lists(edge_dict)
        graph = graph.to_undirected()
        nodes = list(graph.nodes)
        print("Number of nodes ", len(graph.nodes))
        print("Number of edges ", len(graph.edges))

        for node in nodes:
            if node in edge_dict:
                continue
            neighbors = list(graph.neighbors(node))
            graph.remove_node(node)
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[i] == neighbors[j]:
                        continue
                    #if not graph.has_edge(neighbors[i], neighbors[j]):
                    graph.add_edge(neighbors[i], neighbors[j])
                    #else:
                    #    print("Should not happen!!")

        print("Number of nodes ", len(graph.nodes))
        print("Number of edges ", len(graph.edges))
        adj = nx.adjacency_matrix(graph)
        adj.setdiag(0)
        adj = adj.astype(np.float64)
        adj = scipy.sparse.csr_matrix(adj)
        adj.eliminate_zeros()
        edges = list(graph.nodes)

        nx.draw_networkx(graph)
        plt.show()

        with open(self.__edge_adj_file, 'wb') as f:
            pkl.dump(adj, f)
        with open(self.__edges_file, 'wb') as f:
            pkl.dump(edges, f)

    def load_adj_graph(self, do_coarsening=True):
        with open(self.__edge_adj_file, 'rb') as f:
            edges_adj = pkl.load(f)

        if do_coarsening:
            if os.path.exists(self.__perm_file) and os.path.exists(self.__graph_file):
                with open(self.__graph_file, 'rb') as f:
                    graphs = pkl.load(f)
            else:
                graphs, perm = coarsening.coarsen(edges_adj, levels=self.__coarsening_level, self_connections=False)
                with open(self.__perm_file, 'wb') as f:
                    pkl.dump(perm, f)
                with open(self.__graph_file, 'wb') as f:
                    pkl.dump(graphs, f)
        else:
            graphs = [edges_adj]*self.__coarsening_level

        return graphs, edges_adj.shape[0]

    def switch_timestamps(self):
        for i in range(5):
            with open(self.__train_base_dir + "X_training_{}.pickle".format(i), 'rb') as f:
                x_train = pkl.load(f)
            with open(self.__test_base_dir + "X_testing_{}.pickle".format(i), 'rb') as f:
                x_test = pkl.load(f)
            x_train = x_train[:, [3, 2, 1, 0], :, :]
            x_test = x_test[:, [3, 2, 1, 0], :, :]
            with open(self.__test_base_dir + "X_testing_switched_{}.pickle".format(i), 'wb') as f:
                pkl.dump(x_test, f)
            with open(self.__train_base_dir + "X_training_switched_{}.pickle".format(i), 'wb') as f:
                pkl.dump(x_train, f)

    def coarsening_data(self):
        if not os.path.exists(self.__perm_file):
            self.load_adj_graph()

        if not os.path.exists(self.__train_base_dir + "X_training_0.pickle"):
            self.create_fold_partitions()

        with open(self.__perm_file, 'rb') as f:
            perm = pkl.load(f)

        for i in range(self.__fold):
            with open(self.__train_base_dir + "X_training_{}.pickle".format(i), 'rb') as f:
                X_train = pkl.load(f)

            with open(self.__test_base_dir + "X_testing_{}.pickle".format(i), 'rb') as f:
                X_test = pkl.load(f)

            X_train_coarsed = coarsening.perm_data_hist_hist(X_train, perm)
            X_test_coarsed = coarsening.perm_data_hist_hist(X_test, perm)

            with open(self.__train_base_dir + "X_training_coarsed_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_train_coarsed, f)
            with open(self.__test_base_dir + "X_testing_coarsed_{}.pickle".format(i), 'wb') as f:
                pkl.dump(X_test_coarsed, f)

    def create_context(self):
        with open(self.__all_dataset_pickle, 'rb') as f:
            all_dataset = pkl.load(f)

        all_dataset.set_index(["time"], inplace=True)

        test_intervals = [(pd.datetime(year=2015, month=6, day=1), pd.datetime(year=2015, month=6, day=10)),
                          (pd.datetime(year=2015, month=6, day=10), pd.datetime(year=2015, month=6, day=19)),
                          (pd.datetime(year=2015, month=6, day=19), pd.datetime(year=2015, month=6, day=28)),
                          (pd.datetime(year=2015, month=6, day=28), pd.datetime(year=2015, month=7, day=7)),
                          (pd.datetime(year=2015, month=6, day=7), pd.datetime(year=2015, month=6, day=16))]

        time_index = set(all_dataset.time_index.values)
        receptive_field = {}
        for element in time_index:
            if element - 1 in receptive_field:
                receptive_field[element] = receptive_field[element - 1] + 1
            else:
                receptive_field[element] = 1

        i = 0
        for test_index_init, test_index_end in test_intervals:
            mask = ((all_dataset.index >= test_index_init) & (all_dataset.index < test_index_end))
            X_train = all_dataset.loc[np.logical_not(mask)]
            X_test = all_dataset.loc[mask]

            train_data_context = self.convert_data_context(X_train, receptive_field)
            test_data_context = self.convert_data_context(X_test, receptive_field)
            with open(self.__train_base_dir+"X_training_context_{}.pickle".format(i), 'wb') as f:
                pkl.dump(train_data_context, f)
            with open(self.__test_base_dir+"X_testing_context_{}.pickle".format(i), 'wb') as f:
                pkl.dump(test_data_context, f)
            i += 1

    def convert_data_context(self, df_array, receptive_field):
        data_context = []
        N = df_array.shape[0]
        max_time = df_array['time_index'].max()

        for i in range(self.__receptive_field-1, N):

            current_time = df_array.iloc[i]['time_index']
            current_day = df_array.iloc[i]['dayofweek']
            object_i = []

            if receptive_field[current_time] < self.__receptive_field:
                continue

            for r in range(self.__receptive_field - 1, -1, -1):
                timeofday = np.zeros((max_time+1, ))
                dayofweek = np.zeros((7, ))
                timeofday[(current_time-r)%max_time] = 1.0
                previous_day_week = current_day-1 if ((current_time-i)%max_time)-current_time > 0 else current_day
                dayofweek[previous_day_week] = 1.0
                context = np.concatenate([dayofweek, timeofday])
                object_i.append(context)

            data_context.append(object_i)

        return np.asarray(data_context)

    def reduce_context(self):
        for fold in range(self.__fold):
            with open(self.__test_base_dir + 'X_testing_context_{}.pickle'.format(fold), 'rb') as f:
                test_contxt = pkl.load(f)
                non_zero = np.nonzero(test_contxt)[2]
                non_zero = np.unique(non_zero)
                all = np.arange(test_contxt.shape[2])
                zeros = np.setdiff1d(all, non_zero)
                test_contxt = np.delete(test_contxt, zeros, axis=2)

            with open(self.__test_base_dir + 'X_testing_context_{}.pickle'.format(fold), 'wb') as f:
                pkl.dump(test_contxt, f)

            with open(self.__train_base_dir + 'X_training_context_{}.pickle'.format(fold), 'rb') as f:
                train_contxt = pkl.load(f)
                non_zero = np.nonzero(train_contxt)[2]
                non_zero = np.unique(non_zero)
                all = np.arange(train_contxt.shape[2])
                zeros = np.setdiff1d(all, non_zero)
                train_contxt = np.delete(train_contxt, zeros, axis=2)

            with open(self.__train_base_dir + 'X_training_context_{}.pickle'.format(fold), 'wb') as f:
                pkl.dump(train_contxt, f)


def do_kdd(rm):
    print(f"Doing KDD {rm}!!!!!!!!!!")
    kdd = LoadKDD(zero_keeping=False, big_threshold=41, random_sec=rm, least_threshold=0.5)
    kdd.create_trips_from_raw_source()
    kdd.create_concat_vel()
    #kdd.create_fold_partitions()
    #kdd.coarsening_data()
    #kdd.switch_timestamps()
    print(f"Done KDD {rm}!!!!!!!!!!")


def do_chengdu(num_nodes, rm):
    print(f"Running CHENGDU {rm}{num_nodes}!!!!!!!!!!")
    chengdu = LoadChengdu(zero_keeping=True, random_sec=rm, least_threshold=0.5, num_nodes=num_nodes)
    chengdu.create_fold_partitions()
    chengdu.switch_timestamps()
    chengdu.coarsening_data()
    print(f"Done CHENGDU {rm}{num_nodes}!!!!!!!!!!")


def multi():
    from multiprocessing import Process
    procs = []
    for rm in [0.5, 0.6, 0.7, 0.8]:
        kdd = Process(target=do_kdd, args=(rm,))
        #chengdu40 = Process(target=do_chengdu, args=(40, rm))
        #chengdu173 = Process(target=do_chengdu, args=(173, rm))
        kdd.start()
        #chengdu40.start()
        #chengdu173.start()
        procs.append(kdd)
        #procs.append(chengdu40)
        #procs.append(chengdu173)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    multi()
    #kdd = LoadKDD(zero_keeping=False, big_threshold=40, random_sec=0.5, least_threshold=0.5)
    #kdd.create_trips_from_raw_source()
    #kdd.create_concat_vel()
    #kdd.create_vel_hist()
    #kdd.create_avg_hist_hour()
    #kdd.create_fold_partitions()
    #kdd.create_context()
    #kdd.coarsening_data()
    #kdd.switch_timestamps()

    #chengdu = LoadChengdu(zero_keeping=False, random_sec=0.5, least_threshold=0.5)
    #chengdu.filter_data()
    #chengdu.build_trajectories_data()
    #chengdu.create_concat_vel()
    #chengdu.create_vel_hist()
    #chengdu.create_avg_hist_hour()
    #chengdu.create_fold_partitions()
    #chengdu.build_edge_adjacency()
    #chengdu.switch_timestamps()
    #chengdu.coarsening_data()
    #chengdu.create_context()
    #chengdu.reduce_context()
    #create_theoretical_mu_std()