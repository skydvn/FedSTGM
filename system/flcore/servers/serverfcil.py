import time
import torch
from flcore.clients.clientfcil import clientFCIL
from flcore.servers.serverbase import Server
from threading import Thread
from utils.model_utils import read_client_data_FCL, read_client_data_FCL_imagenet1k


class FedFCIL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFCIL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        self.encode_model = encode_model                  # Check this
        self.monitor_dataset = Proxy_Data(test_transform) # Check this

    def train(self):

        if self.args.dataset == 'IMAGENET1k':
            N_TASKS = 500
        else:
            N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        print(str(N_TASKS) + " tasks are available")

        for task in range(N_TASKS):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task

                torch.cuda.empty_cache()
                for i in range(len(self.clients)):

                    if self.args.dataset == 'IMAGENET1k':
                        id, train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=task,
                                                                                                classes_per_task=2,
                                                                                                count_labels=True)
                    else:
                        id, train_data, test_data, label_info = read_client_data_FCL(i, self.data,
                                                                                     dataset=self.args.dataset,
                                                                                     count_labels=True, task=task)

                    # update dataset
                    # vassert (self.users[i].id == id)
                    self.clients[i].next_task(train_data, test_data, label_info)  # assign dataloader for new data

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                """
                    L85-L103 FCIL/fl_main.py
                    - model_g -> global_model
                    - proxy_server -> ? 
                    - 
                """

                self.selected_clients = self.select_clients()
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(glob_iter=glob_pool_graditer)

                for client in self.selected_clients:
                    client.train()
                    """
                        L106-110 comes here
                        - returns client.model + client.proto_grad
                        - grad_i in proto_grad? what is the shape of proto_grad?
                        - append to pool_grad
                    """

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters()

                """
                    L121-L124 comes here
                    - proxy_server.dataloader(pool_grad) do for what?
                """

                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

            print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
            print(max(self.rs_test_acc))
            print("\nAverage time cost per round.")
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

            # self.save_results()
            # self.save_global_model()

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientFCIL)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(glob_iter=glob_iter)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(dataset=self.monitor_dataset, shuffle=True, batch_size=64, drop_last=True)
            self.last_perf = 0
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        print(cur_perf)
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)