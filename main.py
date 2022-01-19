import argparse
import csv
import os
import numpy as np


class Game(object):

    def __init__(self, memory_lvl, n_agent, random_seed=22, verbose=True) -> None:
        super().__init__()
        np.random.seed(random_seed)
        self.verbose = verbose

        print("MEMORY {} | N_AGENT {} | RANDOM SEED {}".format(memory_lvl, n_agent, random_seed))

        # number of rounds a fact will be remembered
        self.memory_lvl = memory_lvl
        # number of agents
        self.n = n_agent
        # number of facts
        self.k = 1 

        # 1. intersection board
        self.inter = np.full((self.n, self.n), 1)
        # 2. facts -> agents in set
        self.fact_2_ag = [set(range(self.n))]
        # 3. agents -> facts in set
        self.ags = [{0} for _ in range(self.n)]
        # 4. alarm for each fact to expire
        self.ag_fact_alarm = [[self.memory_lvl] for _ in range(self.n)] # alarm rings at round n
        self.alarm_queue = [(ag, fact, self.memory_lvl) for ag in range(self.n) for fact in range(self.k)]
    

    def inter_minus(self, ag1: int, ag2:int) -> None:
        self.inter[ag1][ag2] -= 1
        if ag1 != ag2:
            self.inter[ag2][ag1] -= 1
    

    def inter_add(self, ag1: int, ag2: int) -> None:
        self.inter[ag1][ag2] += 1
        if ag1 != ag2:
            self.inter[ag2][ag1] += 1
    

    def simulate(self, n_iter: int, log_v=100):
        for iter_idx in range(n_iter):
            if iter_idx % log_v == 0 and self.verbose:
                print("iter {}/{}".format(iter_idx, n_iter))

            # forget
            while self.alarm_queue[0][2] <= iter_idx:
                ag_idx, fact_idx, iter_alarm = self.alarm_queue.pop(0)
                if iter_alarm == self.ag_fact_alarm[ag_idx][fact_idx]:
                    for ag_tmp in self.fact_2_ag[fact_idx]:
                        self.inter_minus(ag_idx, ag_tmp)
                    self.fact_2_ag[fact_idx].remove(ag_idx)
                    self.ags[ag_idx].remove(fact_idx)
                    self.ag_fact_alarm[ag_idx][fact_idx] = -1

            ag_order = np.arange(self.n)
            np.random.shuffle(ag_order)
            for ag_idx in ag_order:
                # select an agent to interact with
                total_s_facts = sum([len(self.fact_2_ag[fact_idx]) for fact_idx in self.ags[ag_idx]])
                p_arr = self.inter[ag_idx]/total_s_facts
                p_arr[np.argmax(p_arr)] += 1-np.sum(p_arr) # adjust to make its sum be exactly 1.0
                # print(p_arr, np.sum(p_arr))
                # assert np.sum(p_arr) == 1
                ag2_idx = np.random.choice(np.arange(self.n), p=p_arr)
                
                # select a choice
                fact_choice = np.array(list(set.union(self.ags[ag_idx], self.ags[ag2_idx])))
                if ag_idx != ag2_idx:
                    fact_choice = np.append(fact_choice, self.k)
                fact_idx = np.random.choice(fact_choice)

                # print("iter {} | ({}, {}) fact {}".format(iter_idx, ag_idx, ag2_idx, fact_idx))
                next_alarm = iter_idx + self.memory_lvl + 1
                if ag_idx != ag2_idx:
                    if fact_idx == self.k:
                        # create new facts
                        self.k += 1
                        self.inter_add(ag_idx, ag2_idx)
                        self.inter_add(ag_idx, ag_idx)
                        self.inter_add(ag2_idx, ag2_idx)
                        self.fact_2_ag.append(set([ag_idx, ag2_idx]))
                        for fact_alarm in self.ag_fact_alarm:
                            fact_alarm.append(-1)
                    else:
                        if fact_idx not in self.ags[ag_idx]:
                            for ag_tmp in self.fact_2_ag[fact_idx]:
                                self.inter_add(ag_idx, ag_tmp)
                            self.inter_add(ag_idx, ag_idx)
                            self.fact_2_ag[fact_idx].add(ag_idx)
                        elif fact_idx not in self.ags[ag2_idx]:
                            for ag_tmp in self.fact_2_ag[fact_idx]:
                                self.inter_add(ag2_idx, ag_tmp)
                            self.inter_add(ag2_idx, ag2_idx)
                            self.fact_2_ag[fact_idx].add(ag2_idx)

                    self.ags[ag_idx].add(fact_idx)
                    self.ags[ag2_idx].add(fact_idx)
                    self.alarm_queue.append((ag_idx, fact_idx, next_alarm))
                    self.alarm_queue.append((ag2_idx, fact_idx, next_alarm))
                    self.ag_fact_alarm[ag_idx][fact_idx] = next_alarm
                    self.ag_fact_alarm[ag2_idx][fact_idx] = next_alarm
                else:
                    self.alarm_queue.append((ag_idx, fact_idx, next_alarm))
                    self.ag_fact_alarm[ag_idx][fact_idx] = next_alarm
        
        if self.verbose:
            self.print_board()
    

    def print_board(self):
        print("person-fact board")
        for fact_idx in range(self.k):
            for ag_idx in range(self.n):
                print("1 " if ag_idx in self.fact_2_ag[fact_idx] else "0 ", end="")
            print()
        print("intersection board")
        print(self.inter)
        print("====================")
    

    def get_result_indicators(self):
        # 1. Cultural homogeneity
        s_sum = 0
        for i in range(self.n):
            for j in range(i, self.n):
                s_sum += self.inter[i][j]
        culture = s_sum / (self.k*self.n*(self.n-1)/2)

        # 2 & 3. social differentiation & group size
        group_list = self.get_group_list()
        social_diff = len(group_list)
        group_size = sum(group_list) / len(group_list)

        return (culture, social_diff, group_size)
        
    def get_group_list(self) -> list:
        group_list = list()
        self.visited = np.zeros(self.n)
        for i in range(self.n):
            if not self.visited[i]:
                group_list.append(self.search_group(i))
        return group_list

    def search_group(self, root) -> int:
        """ DFS. """
        if self.visited[root]:
            return 0
        
        self.visited[root] = 1
        n_descendant = 0
        for i in range(self.n):
            if self.inter[root][i] and root != i and not self.visited[i]:
                n_descendant += self.search_group(i)
        return n_descendant + 1
        

        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--memory_lvl", type=int, default=4,
    #     help="memory level.")
    
    MEMORY_LVL = [3, 4, 5]
    SYSTEM_SIZE = [6, 50, 100]
    N_ITER = 500
    N_TRAIL = 60

    out_path = os.path.join(os.getcwd(), "output.csv")
    if os.path.isfile(out_path):
        out_file = open(out_path, "a", newline="")
        out_writer = csv.writer(out_file)
    else:
        out_file = open(out_path, "w", newline="")
        out_writer = csv.writer(out_file)
        headers = ["memory level", "system size", "indicators", "n_trail"]
        headers += ["mean", "std"]
        headers += ["trail {}".format(i) for i in range(N_TRAIL)]
        out_writer.writerow(headers)

    for m in MEMORY_LVL:
        for n in SYSTEM_SIZE:
            results = [[], [], []]
            for trail_ctr in range(N_TRAIL):
                game = Game(m, n, random_seed=trail_ctr, verbose=False)
                game.simulate(N_ITER)
                cul, diff, avg_gp_size = game.get_result_indicators()
                results[0].append(cul)
                results[1].append(diff)
                results[2].append(avg_gp_size)
            out_writer.writerow([m, n, "cultural homogeneity", N_TRAIL, \
                np.mean(np.array(results[0])), np.std(np.array(results[0]))] + results[0])
            out_writer.writerow([m, n, "social differentiation", N_TRAIL, \
                np.mean(np.array(results[1])), np.std(np.array(results[1]))] + results[1])
            out_writer.writerow([m, n, "group size", N_TRAIL, \
                np.mean(np.array(results[2])), np.std(np.array(results[2]))] + results[2])



        
