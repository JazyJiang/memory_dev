import os
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd
import ipdb

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        # self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        if self.args.special_token_for_answer:
            self.new_tokens.append("|start_of_answer|")
        return self.new_tokens
    
    def get_codebook_statistics(self):
        code_set = [set() for _ in range(len(self.indices["0"]))] # e.g., identifier length = 4
        for index in self.indices.values():
            for idx, code in enumerate(index):
                code_set[idx].add(code)
        for index in range(len(self.indices["0"])):
            print(f"new token size {index}: {len(code_set[index])}")
        return [len(code_set[_]) for _ in range(4)]
        

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items
        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))
        return self.all_items
    
    def get_warm_items(self):
        self.warm_items = set()
        warm_items = np.load(os.path.join(self.data_path, "warm_item.npy"), allow_pickle=True).tolist()
        for i_id in warm_items:
            self.warm_items.add("".join(self.indices[str(i_id)]))
        return self.warm_items

    def get_cold_items(self):
        self.cold_items = set()
        cold_items = np.load(os.path.join(self.data_path, "cold_item.npy"), allow_pickle=True).tolist()
        for i_id in cold_items:
            self.cold_items.add("".join(self.indices[str(i_id)]))
        return self.cold_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError

    def slice_data(self, n_sample):
        self.inter_data = self.inter_data[:n_sample]



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train", sample_num=-1):
        super().__init__(args)
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.mode = mode
        self.prompt = "What {dataset} products would user be likely to purchase next after buying {dataset} items {history} ?"
        self.prompt = "{history}"
        
        self.special_token_for_answer = None
        self.sample_num = sample_num


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == "test_warm":
            self.warm_items = self.get_warm_items()
            self.inter_data = self._process_test_warm_data()
        elif self.mode == "test_cold":
            self.cold_items = self.get_cold_items()
            self.inter_data = self._process_test_cold_data()
        else:
            raise NotImplementedError


    def _load_data(self):
        self.train_data = np.load(os.path.join(self.data_path, "training_dict.npy"), allow_pickle=True).item()
        self.valid_data = np.load(os.path.join(self.data_path, "validation_dict.npy"), allow_pickle=True).item()
        self.test_data = np.load(os.path.join(self.data_path, "testing_dict.npy"), allow_pickle=True).item()

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_train = dict()
        for uid, items in self.train_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_train[uid] = new_items
        
        self.remapped_valid = dict()
        for uid, items in self.valid_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_valid[uid] = new_items

        self.remapped_test = dict()
        for uid, items in self.test_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_test[uid] = new_items

    def _process_train_data(self):

        inter_data = []
        for uid in self.remapped_train:
            items = self.remapped_train[uid]# input of each training sample
            if len(items)>1: # a training user should at least have two interactions
                if self.args.subseq:
                    for i in range(1, len(items)):
                        one_data = dict()
                        # one_data["user"] = uid
                        one_data["item"] = items[i]
                        history = items[:i]
                        if self.max_his_len > 0:
                            history = history[-self.max_his_len:]
                        if self.add_prefix:
                            history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                        one_data["inters"] = self.prompt.format(history="".join(history)) 
                        inter_data.append(one_data)
                else:
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = items[:-1]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(history="".join(history)) 
                    inter_data.append(one_data)
        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_valid:
            items = self.remapped_valid[uid]
            train_items = self.remapped_train[uid]
            if len(items):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[0]
                history = train_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.prompt.format(history="".join(history)) 
                inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items
                history = train_items + valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                # one_data["inters"] = "".join(history) + self.special_token_for_answer
                # one_data["inters"] = self.prompt.format(dataset=self.dataset,history="".join(history)) + self.special_token_for_answer
                one_data["inters"] = self.prompt.format(history="".join(history)) 
                inter_data.append(one_data)

        if self.sample_num > 0:
            # all_inter_idx = range(len(inter_data))
            # sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # inter_data = np.array(inter_data)[sample_idx].tolist()
            inter_data = inter_data[:self.sample_num]

        return inter_data
    

    def _process_test_warm_data(self):
        warm_cnt = 0
        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                gold = []
                for item in items:
                    if item in self.warm_items:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = train_items + valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                # one_data["inters"] = "".join(history) 
                # one_data["inters"] = self.prompt.format(dataset=self.dataset,history="".join(history)) + self.special_token_for_answer
                one_data["inters"] = self.prompt.format(history="".join(history)) 
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("warm interaction in test:", warm_cnt)
        return inter_data
    

    def _process_test_cold_data(self):
        inter_data = []
        cold_cnt = 0
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                gold = []
                for item in items:
                    if item in self.cold_items: #TODO
                        gold.append(item)
                        cold_cnt += 1
                one_data["item"] = gold
                history = train_items + valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                # one_data["inters"] = "".join(history)
                # one_data["inters"] = self.prompt.format(dataset=self.dataset,history="".join(history)) + self.special_token_for_answer
                one_data["inters"] = self.prompt.format(history="".join(history)) 
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        print("cold interaction in test:", cold_cnt)
        return inter_data
    
    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])
    
import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SeqRecDatasetCSV(BaseDataset):
    def __init__(self, args, mode="train", sample_num=-1, 
                 train_file=None, valid_file=None, test_file=None):
        super().__init__(args)
        self.data_path = args.data_path
        self.dataset = getattr(args, "dataset", "default")
        self.index_file = getattr(args, "index_file", "_index.json")

        self.mode = mode
        self.prompt = "{history}"
        self.special_token_for_answer = None
        self.sample_num = sample_num

        # read csv file path
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.test_file = args.test_file

        # load interaction and remap semantic iD
        self._load_from_csv()
        self._remap_items()

        # load interaction data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == "test_warm":
            self.warm_items = self.get_warm_items()
            self.inter_data = self._process_test_warm_data()
        elif self.mode == "test_cold":
            self.cold_items = self.get_cold_items()
            self.inter_data = self._process_test_cold_data()
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    # =====================
    #       load csv interacitions
    # =====================
    def _load_from_csv(self):
        def load_split_file(default_dir, given_path, split_name):
            if given_path is not None:
                print(f"✅ Using specified {split_name} file: {given_path}")
                return pd.read_csv(given_path)
            else:
                for filename in os.listdir(default_dir):
                    if self.dataset in filename and filename.endswith(".csv"):
                        path = os.path.join(default_dir, filename)
                        print(f"✅ Found {split_name} file: {path}")
                        return pd.read_csv(path)
                raise FileNotFoundError(f"No {split_name} file found in {default_dir}")

        # read train / valid / test
        self.train_data = load_split_file(os.path.join(self.data_path, "train"), self.train_file, "train")
        self.valid_data = load_split_file(os.path.join(self.data_path, "valid"), self.valid_file, "valid")
        self.test_data  = load_split_file(os.path.join(self.data_path, "test"),  self.test_file,  "test")

        # read semantic ID file
        info_path = os.path.join(self.data_path, "info", f"{self.dataset}{self.index_file}")
        with open(info_path, 'r') as f:
            self.indices = json.load(f)
        self.identifier_len = len(list(self.indices.values())[0])

    # =====================
    #       item remap
    # =====================
    def _remap_items(self):
        def remap_split(df):
            remap_dict = {}
            for index, row in df.iterrows():
                history_ids = eval(row["history_item_id"])
                target_code = "".join(self.indices[str(row["item_id"])])
                history_code = ["".join(self.indices[str(i)]) for i in history_ids]
                remap_dict[index] = [history_code, target_code]
            return remap_dict

        self.remapped_train = remap_split(self.train_data)
        self.remapped_valid = remap_split(self.valid_data)
        self.remapped_test  = remap_split(self.test_data)

    # =====================
    #       warm/cold sets
    # =====================
    def get_warm_items(self):
        warm_path = os.path.join(self.data_path, "info", f"{self.dataset}_warm_item.npy")
        warm_items = np.load(warm_path, allow_pickle=True).tolist()
        return {"".join(self.indices[str(i)]) for i in warm_items}

    def get_cold_items(self):
        cold_path = os.path.join(self.data_path, "info", f"{self.dataset}_cold_item.npy")
        cold_items = np.load(cold_path, allow_pickle=True).tolist()
        return {"".join(self.indices[str(i)]) for i in cold_items}

    # =====================
    #       data pre-processing
    # =====================
    def _process_train_data(self):
        inter_data = []
        for uid, (history, target) in self.remapped_train.items():
            one_data = dict(item=target, inters=self.prompt.format(history="".join(history)))
            inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid, (history, target) in self.remapped_valid.items():
            one_data = dict(item=target, inters=self.prompt.format(history="".join(history)))
            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid, (history, target) in self.remapped_test.items():
            one_data = dict(item=[target], inters=self.prompt.format(history="".join(history)))
            inter_data.append(one_data)
        print(f"interaction in test: {len(inter_data)}")
        if self.sample_num > 0:
            inter_data = inter_data[:self.sample_num]
        return inter_data

    def _process_test_warm_data(self):
        inter_data = []
        warm_cnt = 0
        for uid, (history, target) in self.remapped_test.items():
            one_data = dict()
            if target in self.warm_items:
                one_data["item"] = [target]
                warm_cnt += 1
            else:
                one_data["item"] = []
            one_data["inters"] = self.prompt.format(history="".join(history))
            inter_data.append(one_data)
        print("warm interaction in test:", warm_cnt)
        return inter_data

    def _process_test_cold_data(self):
        inter_data = []
        cold_cnt = 0
        for uid, (history, target) in self.remapped_test.items():
            one_data = dict()
            if target in self.cold_items:
                one_data["item"] = [target]
                cold_cnt += 1
            else:
                one_data["item"] = []
            one_data["inters"] = self.prompt.format(history="".join(history))
            inter_data.append(one_data)
        print("cold interaction in test:", cold_cnt)
        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])


# class SeqRecDatasetCSV(BaseDataset):
#     def __init__(self, args, mode="train", sample_num=-1):
#         super().__init__(args)

#         self.data_path = args.data_path

#         self.mode = mode
#         self.prompt = "What {dataset} products would user be likely to purchase next after buying {dataset} items {history} ?"
#         self.prompt = "{history}"
        
#         self.special_token_for_answer = None
#         self.sample_num = sample_num

#         # load data
#         self._load_from_csv()
#         self._remap_items()
        
#         # load data
#         if self.mode == 'train':
#             self.inter_data = self._process_train_data()
#         elif self.mode == 'valid':
#             self.inter_data = self._process_valid_data()
#         elif self.mode == 'test':
#             self.inter_data = self._process_test_data()
#         elif self.mode == "test_warm":
#             self.warm_items = self.get_warm_items()
#             self.inter_data = self._process_test_warm_data()
#         elif self.mode == "test_cold":
#             self.cold_items = self.get_cold_items()
#             self.inter_data = self._process_test_cold_data()
#         else:
#             raise NotImplementedError


#     def _load_from_csv(self):
#         for filename in os.listdir(os.path.join(self.data_path, "train")):
#             if self.dataset in filename and ".csv" in filename:
#                 self.train_data = pd.read_csv(os.path.join(self.data_path, "train", filename))

#         for filename in os.listdir(os.path.join(self.data_path, "valid")):
#             if self.dataset in filename and ".csv" in filename:
#                 self.valid_data = pd.read_csv(os.path.join(self.data_path, "valid", filename))

#         for filename in os.listdir(os.path.join(self.data_path, "test")):
#             if self.dataset in filename and ".csv" in filename:
#                 self.test_data = pd.read_csv(os.path.join(self.data_path, "test", filename))

#         with open(os.path.join(self.data_path, "info", f"{self.dataset}{self.index_file}"), 'r') as f:
#             self.indices = json.load(f)

#         self.identifier_len = len(list(self.indices.values())[0])

#     def _remap_items(self):
#         self.remapped_train = dict()
#         for index, row in self.train_data.iterrows():
#             uid = index
#             history_ids = eval(row["history_item_id"])
#             target_code = "".join(self.indices[str(row["item_id"])])
#             history_code = ["".join(self.indices[str(i)]) for i in history_ids]
#             self.remapped_train[uid] = [history_code, target_code]
        
#         self.remapped_valid = dict()
#         for index, row in self.valid_data.iterrows():
#             uid = index
#             history_ids = eval(row["history_item_id"])
#             target_code = "".join(self.indices[str(row["item_id"])])
#             history_code = ["".join(self.indices[str(i)]) for i in history_ids]
#             self.remapped_valid[uid] = [history_code, target_code]

#         self.remapped_test = dict()
#         for index, row in self.test_data.iterrows():
#             uid = index
#             history_ids = eval(row["history_item_id"])
#             target_code = "".join(self.indices[str(row["item_id"])])
#             history_code = ["".join(self.indices[str(i)]) for i in history_ids]
#             self.remapped_test[uid] = [history_code, target_code]

#     def get_warm_items(self):
#         self.warm_items = set()
#         warm_items = np.load(os.path.join(self.data_path, "info", f"{self.dataset}_warm_item.npy"), allow_pickle=True).tolist()
#         for i_id in warm_items:
#             self.warm_items.add("".join(self.indices[str(i_id)]))
#         return self.warm_items

#     def get_cold_items(self):
#         self.cold_items = set()
#         cold_items = np.load(os.path.join(self.data_path, "info", f"{self.dataset}_cold_item.npy"), allow_pickle=True).tolist()
#         for i_id in cold_items:
#             self.cold_items.add("".join(self.indices[str(i_id)]))
#         return self.cold_items
    
#     def _process_train_data(self):
#         inter_data = []
#         for uid in self.remapped_train:
#             history, target = self.remapped_train[uid]# input of each training sample
#             one_data = dict()
#             one_data["item"] = target
#             one_data["inters"] = self.prompt.format(history="".join(history)) 
#             inter_data.append(one_data)
#             # ipdb.set_trace()
#         return inter_data
    
#     def _process_valid_data(self):
#         inter_data = []
#         for uid in self.remapped_valid:
#             history, target = self.remapped_valid[uid]
#             one_data = dict()
#             one_data["item"] = target
#             one_data["inters"] = self.prompt.format(history="".join(history)) 
#             inter_data.append(one_data)
#         return inter_data

#     def _process_test_data(self):
#         inter_data = []
#         for uid in self.remapped_test:
#             history, target = self.remapped_test[uid]
#             one_data = dict()
#             one_data["item"] = [target]
#             one_data["inters"] = self.prompt.format(history="".join(history)) 
#             inter_data.append(one_data)

#         print("interaction in test:", len(inter_data))
#         if self.sample_num > 0:
#             inter_data = inter_data[:self.sample_num]

#         return inter_data
    
#     def _process_test_warm_data(self):
#         warm_cnt = 0
#         inter_data = []
#         for uid in self.remapped_test:
#             history, target = self.remapped_test[uid]
#             one_data = dict()
#             gold = []
#             if target in self.warm_items:
#                 gold.append(target)
#                 warm_cnt += 1
#             one_data["item"] = gold
#             one_data["inters"] = self.prompt.format(history="".join(history))
#             inter_data.append(one_data)
#         print("warm interaction in test:", warm_cnt)

#         if self.sample_num > 0:
#             inter_data = inter_data[:self.sample_num]

#         return inter_data
    
#     def _process_test_cold_data(self):
#         inter_data = []
#         cold_cnt = 0
#         for uid in self.remapped_test:
#             history, target = self.remapped_test[uid]
#             one_data = dict()
#             gold = []
#             if target in self.cold_items:
#                 gold.append(target)
#                 cold_cnt += 1
#             one_data["item"] = gold
#             one_data["inters"] = self.prompt.format(history="".join(history)) 
#             inter_data.append(one_data)
#         print("cold interaction in test:", cold_cnt)

#         if self.sample_num > 0:
#             inter_data = inter_data[:self.sample_num]

#         return inter_data

#     def __len__(self):
#         return len(self.inter_data)

#     def __getitem__(self, index):
#         d = self.inter_data[index]
#         return dict(input_ids=d["inters"], labels=d["item"])


