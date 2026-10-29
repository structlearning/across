import heapq
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import psutil
import torch
from loguru import logger


def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return x.cuda() if av.has_cuda and av.want_cuda else x


def check_pretrained_hashcodes(av):
    ...


def user_time():
    return psutil.Process().cpu_times().user


# psutil.Process().cpu_times().user
# time.process_time()
# time.time()
def optimized_cosine_similarity(a, b1):
    r"""
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the cosine similarity cos(a[i],b[j]) for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    eps = 1e-8
    return (a @ b1.T) / (
        (np.linalg.norm(a, axis=1)[:, None] + eps) * (np.linalg.norm(b1, axis=1)[None, :] + eps)
    )


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_asym_sim(av, query_embeds, corpus_embeds):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    # sim =  np.sum(av.T - np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sig_in = np.sum(np.maximum(0, query_embeds - corpus_embeds), axis=1)
    sig_out = np_sigmoid(av.sigmoid_a * sig_in + av.sigmoid_b)
    return sig_out


def dot_sim(a, b):
    r"""
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the dot similarity a[i]*b[j] for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    return (a @ b.T)  # (a[:,None,:]*b[None,:,:]).sum(-1) #


def asym_sim(av, query_embeds, corpus_embeds):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    # sim =  np.sum(av.T - np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sim = -np.sum(np.maximum(0, query_embeds - corpus_embeds), axis=1)
    return sim


def wjac_sim(a, b):
    '''
        query_embeds: 1 x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    return np.minimum(a, b).sum(-1) / (np.maximum(a, b).sum(-1) + 1e-8)


def fetch_gaussian_hyperplanes(av, fmap_dim):
    """
        Based on the dimention of the feature map
        Load/generate specified no of gaussian hyperplanes
        av.hcode_dim specifies no of planes to generate
    """
    logger.info('Fetching gaussian hplanes ...')
    fp = f'./data/gauss_hplanes_hdim{av.hcode_dim}_fmap_dim{fmap_dim}.pkl'
    if os.path.exists(fp):
        all_data = pickle.load(open(fp, 'rb'))
        hplanes = all_data['hplanes']
    else:
        hplanes = np.random.normal(size=(fmap_dim, av.hcode_dim))
        all_data = {
            'hplanes': hplanes,
            'fmap_dim': fmap_dim,
            'hcode_dim': av.hcode_dim
        }
        pickle.dump(all_data, open(fp, "wb"))
    logger.info("From %s", fp)
    return hplanes


class LSH:
    """
    Requires the following av args to be specified:
    av.embed_dim  :       query/corpus expected to be of same (known) embedding dim
    av.m_use      :     decide when init
    av.hcode_dim :        decide when init
    av.num_hash_tables :  decide when init
    av.subset_size :      decode when init
                            subsets of hashcode to consider in each hash function
                            dictates no. of buckets
    We store following in LSH object:
    w samples for feature maps: universal for all query/corpus points
    gaussian hplanes: (Universal) dictated by embedding dim, no of w samples ,required hashcode dim
    """
    def __init__(self, av):
        super().__init__()
        self.av = av
        self.hashcode_pickle_fp = self.av.pickle_fp
        if self.hashcode_pickle_fp != "":
            self.hdata = pickle.load(open(self.hashcode_pickle_fp, "rb"))
        self.num_hash_tables = self.av.num_hash_tables
        # No. of buckets in a hashTable is 2^subset_size
        self.subset_size = self.av.subset_size
        assert (self.subset_size <= av.hcode_dim)
        self.powers_of_two = cudavar(self.av, torch.from_numpy(1 << np.arange(self.subset_size - 1, -1, -1)).type(torch.FloatTensor))
        # generates self.hash_functions, containing indices for hash functions
        self.init_hash_functions()
        if self.av.use_pretrained_fmap:
            self.gauss_hplanes_fhash = fetch_gaussian_hyperplanes(self.av, self.av.tr_fmap_dim)
        else:
            self.gauss_hplanes_fhash = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim * self.av.m_use * 4)
        if not self.av.FMAP_SELECT == -1:
            self.gauss_hplanes_fhash = self.gauss_hplanes_fhash[:self.av.FMAP_SELECT, :]

        if self.av.trained_cosine_fmap_pickle_fp != "" or self.av.use_pretrained_fmap:
            self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.av, self.av.tr_fmap_dim)
        else:
            self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim)
        self.gauss_hplanes_dot = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim + 1)
        self.gauss_hplanes_dot2 = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim + 2)

        # Timing methods anf funcs
        self.timing_methods = ["real", "user", "process_time"]
        self.timing_funcs = {"real": time.time, "user": user_time, "process_time": time.process_time}
        # self.mean = None
        self.trained_cosine_fmap_pickle_fp = self.av.trained_cosine_fmap_pickle_fp
        if self.trained_cosine_fmap_pickle_fp != "":
            assert (self.av.pickle_fp == "")
            assert (self.av.HASH_MODE == "cosine")
            fmapdata = pickle.load(open(self.trained_cosine_fmap_pickle_fp, "rb"))
            self.hdata = {}
            batch_sz = 50000
            # Writing split manually to ensure correctness
            batches = []
            for i in range(0, fmapdata['corpus'].shape[0], batch_sz):
                batches.append(fmapdata['corpus'][i:i + batch_sz])
            assert sum([item.shape[0] for item in batches]) == fmapdata['corpus'].shape[0]
            hcode_list = []
            for batch_item in batches:
                projections = batch_item @ self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))

            self.hdata['corpus'] = torch.cat(hcode_list)
            self.hdata['query'] = {}
            # for mode in ["train", "test", "val"]:
            for mode in ["train", "val"]:
                projections = fmapdata['query'][mode].cpu() @ self.gauss_hplanes_cos
                self.hdata['query'][mode] = cudavar(self.av, torch.tensor(np.sign(projections)))

        if self.av.use_pretrained_fmap:
            self.tr_fmap_data, self.fmap_model_weights_np = check_pretrained_fmaps(self.av)
        if self.av.use_pretrained_hcode:
            # if self.av.TASK == "sighinge":
            #    self.tr_hcode_data,self.hcode_model_weights_np = check_pretrained_hashcodes_endtoend(self.av)
            # el
            if self.av.HASH_MODE == "cosine" or self.av.HASH_MODE == "dot" or self.av.HASH_MODE == "dot2" or self.av.HASH_MODE == "flora":
                self.tr_hcode_data, self.hcode_model_weights_np = check_pretrained_hashcodes_hingeemb(self.av)
            elif self.av.HASH_MODE == "fhash":
                self.tr_hcode_data, self.hcode_model_weights_np = check_pretrained_hashcodes(self.av)
            else:
                raise NotImplementedError()

    def generate_fmap(self, av, embeds, isQuery=False):
        ...

    def init_hash_functions(self):
        """
            Each hash function is a random subset of the hashcode.
        """
        self.hash_functions = cudavar(self.av, torch.LongTensor([]))

        hash_code_dim = self.av.hcode_dim
        indices = list(range(hash_code_dim))
        for i in range(self.num_hash_tables):
            random.shuffle(indices)
            self.hash_functions = torch.cat((self.hash_functions, cudavar(self.av, torch.LongTensor([indices[:self.subset_size]]))), dim=0)

    def index_corpus(self, corpus_embeds, hash_mode):
        s = time.time()
        # TODO: may need to do torch conversion of corpus_embeds
        if self.av.HASH_MODE == "wmh":  # and (corpus_embeds[0]==0).all():
            rows_idx = []
            for idx in range(len(corpus_embeds)):
                if all(corpus_embeds[idx] == 0):
                    rows_idx.append(idx)
            self.corpus_embeds = np.delete(corpus_embeds, rows_idx, axis=0)
            # self.corpus_embeds = corpus_embeds[1:]
        else:
            self.corpus_embeds = corpus_embeds

        assert self.av.embed_dim == self.corpus_embeds.shape[1]
        assert self.av.m_use * self.av.embed_dim <= self.av.m_load
        self.M = np.linalg.norm(corpus_embeds, axis=-1).max()

        if self.av.HASH_MODE == "wmh":
            self.corpus_hashcodes = get_wmh_hcode(self.av, self.corpus_embeds)
        else:
            self.corpus_hashcodes = self.fetch_RH_hashcodes(self.corpus_embeds, hash_mode, isQuery=False).cpu()
        assert (self.corpus_embeds.shape[0] == self.corpus_hashcodes.shape[0])
        self.num_corpus_items = self.corpus_embeds.shape[0]
        # generates self.hashcode_mat (containing +1/-1, used for bucketing)
        self.hashcode_mat = self.preprocess_hashcodes(self.corpus_hashcodes)
        # Assigns corpus items to buckets in each of the tables
        # generates dict self.all_hash_tables containing bucketId:courpusItemIDs
        self.bucketify()
        logger.info(f"Corpus indexed. Time taken {time.time() - s:.3f} sec")

    def fetch_RH_hashcodes(self, embeds, hash_mode, isQuery, qid=None):
        """
            embeds: can be 1 or many from query or corpus
        """
        if self.hashcode_pickle_fp != "" or self.trained_cosine_fmap_pickle_fp != "":
            if isQuery == False:
                # print(f"isQuery: {isQuery}  hashcodes shape : {self.hdata['corpus'].double().shape} {self.hdata['corpus'].double().dtype}")
                return cudavar(self.av, self.hdata['corpus'].double())
            # print(f"isQuery: {isQuery}  hashcodes shape : {self.hdata['query'][self.av.SPLIT][qid].double()[None,:].shape} {self.hdata['query'][self.av.SPLIT][qid].double()[None,:].dtype}")
            return cudavar(self.av, self.hdata['query'][self.av.SPLIT][qid].double()[None, :])

        if self.av.SPECIAL_FHASH == "SymCorpus":
            isQuery = False
        elif self.av.SPECIAL_FHASH == "SymQuery":
            isQuery = True
        max_norm = np.max(np.linalg.norm(embeds, axis=1))
        d = embeds.shape[1]
        batch_sz = 50000
        # Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0], batch_sz):
            batches.append(embeds[i:i + batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]
        hcode_list = []
        for batch_item in batches:
            if hash_mode == "cosine":
                if self.av.use_pretrained_hcode:
                    # print("Should be here")
                    if self.hcode_model_weights_np['num_layers'] == 1:
                        projections = batch_item @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        projections = batch_item @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                        for idx in range(1, self.hcode_model_weights_np['num_layers']):
                            projections[projections < 0] = 0
                            projections = projections @ self.hcode_model_weights_np[idx]['np_w'].T + self.hcode_model_weights_np[idx]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                else:
                    # print("Should NOT be here")
                    projections = batch_item @ self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))
            elif hash_mode == "flora":
                # if self.av.use_pretrained_hcode:
                if isQuery:
                    fmaps = batch_item @ self.hcode_model_weights_np['np_w_q'].T + self.hcode_model_weights_np['np_b_q']
                else:
                    fmaps = batch_item @ self.hcode_model_weights_np['np_w_c'].T + self.hcode_model_weights_np['np_b_c']
                assert self.hcode_model_weights_np['num_layers'] == 1
                projections = fmaps @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                projections = np.tanh(self.av.TANH_TEMP * projections)
                # else:
                #    #print("Should NOT be here")
                #    projections = batch_item@self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))
            elif hash_mode == "dot" or hash_mode == "dot2":
                # NOTE: for query it will ultimately append 0, but we may directly do that under if isQuery to speed up
                if hash_mode == "dot":
                    batch_item_scaled = batch_item / max_norm
                    app = np.expand_dims(np.sqrt(1 - np.square(np.linalg.norm(batch_item_scaled, axis=1))), axis=-1)
                    batch_item_augmented = np.hstack((batch_item_scaled, app))
                elif hash_mode == "dot2":
                    zero_np = np.zeros((batch_item.shape[0], 1))
                    app = np.expand_dims(np.sqrt(np.square(self.M) - np.square(np.linalg.norm(batch_item, axis=1))), axis=-1)
                    if isQuery:
                        batch_item_augmented = np.hstack((app, batch_item, zero_np)) / self.M
                    else:
                        batch_item_augmented = np.hstack((zero_np, batch_item, app)) / self.M

                if self.av.use_pretrained_hcode:
                    if self.hcode_model_weights_np['num_layers'] == 1:
                        projections = batch_item_augmented @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        projections = batch_item_augmented @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                        for idx in range(1, self.hcode_model_weights_np['num_layers']):
                            projections[projections < 0] = 0
                            projections = projections @ self.hcode_model_weights_np[idx]['np_w'].T + self.hcode_model_weights_np[idx]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                else:
                    if hash_mode == "dot":
                        projections = batch_item_augmented @ self.gauss_hplanes_dot
                    elif hash_mode == "dot2":
                        projections = batch_item_augmented @ self.gauss_hplanes_dot2

                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))
            elif hash_mode == "fhash":
                # fmaps =  generate_fourier_map(self.av, batch_item, self.ws, self.pdfs, isQuery)
                # fmaps =  generate_fourier_map_tensorized_np(self.av, batch_item, self.ws[:d*self.av.m_use], self.sqrt_pdfs[:d*self.av.m_use], isQuery)
                # if isQuery:
                #    fmaps =  self.generate_query_fmap2(self.av, batch_item/self.av.SCALE)[None,:]
                # else:
                fmaps = self.generate_fmap(self.av, batch_item / self.av.SCALE, isQuery)
                if not self.av.FMAP_SELECT == -1:
                    fmaps = fmaps[:, :self.av.FMAP_SELECT]
                if self.av.use_pretrained_fmap:
                    if self.av.FMAP_LOSS_TYPE == "AsymFmapCos":
                        if isQuery:
                            fmaps = fmaps @ self.fmap_model_weights_np['np_w_q'].T + self.fmap_model_weights_np['np_b_q']
                            fmaps = fmaps / np.linalg.norm(fmaps, axis=-1, keepdims=True)
                            # print("Should be here Query: ", qid)
                            # assert (np.allclose(fmaps,self.tr_fmap_data['query'][self.av.SPLIT].cpu().numpy()[qid],atol=1e-06))
                        else:
                            fmaps = fmaps @ self.fmap_model_weights_np['np_w_c'].T + self.fmap_model_weights_np['np_b_c']
                            fmaps = fmaps / np.linalg.norm(fmaps, axis=-1, keepdims=True)
                            # print("Should be here", fmaps.shape)
                    elif self.av.FMAP_LOSS_TYPE == "FmapCos":
                        fmaps = fmaps @ self.fmap_model_weights_np['np_w'].T + self.fmap_model_weights_np['np_b']
                        fmaps = fmaps / np.linalg.norm(fmaps, axis=-1, keepdims=True)

                if self.av.DEBUG:
                    assert np.all(np.isclose(np.linalg.norm(fmaps[0]),
                                             np.linalg.norm(fmaps, axis=1)))

                assert fmaps.shape[0] == batch_item.shape[0],\
                f"fmaps shape={fmaps.shape[0]}, batch_item shape={batch_item.shape[0]}"
                # if self.mean is None:
                #    self.mean =  np.mean(fmaps,axis=0)
                # fmaps -= self.mean
                if self.av.use_pretrained_hcode:
                    if self.av.E2EVER == "e2e3" or self.av.E2EVER == "e2e4":
                        if self.hcode_model_weights_np['num_layers'] == 1:
                            projections = batch_item @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        else:
                            projections = batch_item @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                            for idx in range(1, self.hcode_model_weights_np['num_layers']):
                                projections[projections < 0] = 0
                                projections = projections @ self.hcode_model_weights_np[idx]['np_w'].T + self.hcode_model_weights_np[idx]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        # projections = batch_item@self.hcode_model_weights_np['np_w'].T+self.hcode_model_weights_np['np_b']
                        # projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        if self.hcode_model_weights_np['num_layers'] == 1:
                            projections = fmaps @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        else:
                            projections = fmaps @ self.hcode_model_weights_np[0]['np_w'].T + self.hcode_model_weights_np[0]['np_b']
                            for idx in range(1, self.hcode_model_weights_np['num_layers']):
                                projections[projections < 0] = 0
                                projections = projections @ self.hcode_model_weights_np[idx]['np_w'].T + self.hcode_model_weights_np[idx]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        # print("Should be here")
                        # projections = fmaps@self.hcode_model_weights_np['np_w'].T+self.hcode_model_weights_np['np_b']
                        # projections = np.tanh(self.av.TANH_TEMP * projections)
                        # print("Shoulf not be here")
                else:
                    # print("Should be here")
                    projections = fmaps @ self.gauss_hplanes_fhash

                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))
            elif hash_mode == "random":
                random_hash_codes = np.random.choice([-1, 1], size=(batch_item.shape[0], self.av.hcode_dim))
                hcode_list.append(cudavar(self.av, torch.tensor(random_hash_codes, dtype=float)))
            elif hash_mode == "fhash_sym":
                fmaps = self.generate_fmap(self.av, batch_item / self.av.SCALE, True)
                assert fmaps.shape[0] == batch_item.shape[0],\
                f"fmaps shape={fmaps.shape[0]}, batch_item shape={batch_item.shape[0]}"
                projections = fmaps @ self.gauss_hplanes_fhash
                hcode_list.append(cudavar(self.av, torch.tensor(np.sign(projections))))
            else:
                # Can be thrown if hash_mode=None for no_bucket and somehow hashcodes are being generated for query
                raise NotImplementedError()
        hashcodes = torch.cat(hcode_list)
        # print(f"isQuery: {isQuery}  hashcodes shape : {hashcodes.shape}  {hashcodes.dtype} ")
        return hashcodes

    def preprocess_hashcodes(self, all_hashcodes):
        if self.av.HASH_MODE == "wmh":
            return all_hashcodes
        all_hashcodes = cudavar(self.av, torch.sign(all_hashcodes))
        if (torch.sign(all_hashcodes) == 0).any():
            logger.info("Hashcode had 0 bits. replacing all with 1")
            all_hashcodes[all_hashcodes == 0] = 1
        return all_hashcodes

    def assign_bucket(self, function_id, node_hash_code):
        func = self.hash_functions[function_id]

        if self.av.HASH_MODE == "wmh":
            return '_'.join(node_hash_code[func])

        # convert sequence of -1 and 1 to binary by replacing -1 s to 0
        binary_id = torch.max(torch.index_select(node_hash_code, dim=0, index=func), cudavar(self.av, torch.LongTensor([0])))
        # map binary sequence to int which is bucket Id
        bucket_id = self.powers_of_two @ binary_id.type(torch.FloatTensor).to(self.powers_of_two)
        return bucket_id.item()

    def bucketify(self):
        """
          For all hash functions:
            Loop over all corpus items
              Assign corpus item to bucket in hash table corr. to hash function
        """
        s = time.time()
        self.all_hash_tables = []
        for func_id in range(self.num_hash_tables):
            hash_table = defaultdict(list)  # {}
            # for idx in range(2**self.subset_size):
            #    hash_table[idx] = []
            for item in range(self.num_corpus_items):
                hash_table[self.assign_bucket(func_id, self.hashcode_mat[item])].append(item)
            self.all_hash_tables.append(hash_table)

    def pretty_print_hash_tables(self, topk):
        """
            I've found this function useful to visualize corpus distribution across buckets
        """
        for table_id in range(self.num_hash_tables):
            if self.av.HASH_MODE == "wmh":
                len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in self.all_hash_tables[table_id].keys()])[::-1][:topk]
            else:
                len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in range(2**self.subset_size)])[::-1][:topk]
            len_list_str = [str(i) for i in len_list]
            lens = '|'.join(len_list_str)
            print(lens)

    def heapify(self, q_embed, candidate_list, K, use_tensor=False, qid=0):
        """
            use q_embed , candidate_list, corpus_embeds to fetch top K items
        """
        time_dict = {tm: {} for tm in self.timing_methods}
        other_data_dict = {}
        # if use_tensor then parallelize scoring, else GPU
        # TODO: discuss if we want to do the GPU vs non-GPU thing
        # If no, then torch can be avoided completely
        # If yes, we need to use torch instead of np below
        if len(candidate_list) == 0:
            for k in time_dict:
                time_dict[k]['score_computation_time'] = 0.0
                time_dict[k]['heap_procedure_time'] = 0.0
                time_dict[k]['take_time'] = 0.0
                time_dict[k]['sort_procedure_time'] = 0.0
            other_data_dict['len_candidate_list'] = 0
            return list(), list(), time_dict, other_data_dict
        # print(len(candidate_list),self.corpus_embeds.shape)
        score_timer_start = {}
        for tm in self.timing_methods:
            score_timer_start[tm] = self.timing_funcs[tm]()

        # s = time.time()
        take_start = {}
        for tm in self.timing_methods:
            take_start[tm] = self.timing_funcs[tm]()
        candidate_corpus_embeds = np.take(self.corpus_embeds, candidate_list, axis=0)
        for tm in self.timing_methods:
            time_dict[tm]['take_time'] = self.timing_funcs[tm]() - take_start[tm]
        # print("DEBUG score::",time.time()-s)
        # s = time.time()
        if self.av.HASH_MODE == 'flora':
            candidate_corpus_embeds = np.take(self.cembeds, candidate_list, axis=0)
            # scores = asym_sim(self.av, self.qembeds[qid],candidate_corpus_embeds)
            scores = sigmoid_asym_sim(self.av, self.qembeds[qid], candidate_corpus_embeds)
        elif self.av.TASK == 'hinge':
            # self.pretty_print_hash_tables(15)
            # exit(1)
            scores = asym_sim(self.av, q_embed, candidate_corpus_embeds)
        elif self.av.TASK == 'sighinge':
            # self.pretty_print_hash_tables(15)
            # exit(1)
            scores = sigmoid_asym_sim(self.av, q_embed, candidate_corpus_embeds)
        elif self.av.TASK == 'dot':
            # self.pretty_print_hash_tables(15)
            # exit(1)
            scores = dot_sim(q_embed, candidate_corpus_embeds).reshape(-1)
        elif self.av.TASK == 'wjac':
            # self.pretty_print_hash_tables(15)
            # exit(1)
            scores = wjac_sim(q_embed, candidate_corpus_embeds).reshape(-1)
        else:
            # self.pretty_print_hash_tables(15)
            # exit(1)
            scores = optimized_cosine_similarity(q_embed, candidate_corpus_embeds).reshape(-1)
        # print("DEBUG score::",time.time()-s)
        for tm in self.timing_methods:
            time_dict[tm]['score_computation_time'] = self.timing_funcs[tm]() - score_timer_start[tm]
        heap_timer_start = {}
        for tm in self.timing_methods:
            heap_timer_start[tm] = self.timing_funcs[tm]()
        # s = time.time()
        if K >= 0:
            score_heap = []
            heap_size = 0

            # print("DEBUG heap::",time.time()-s)
            # s = time.time()

            for i in range(len(candidate_list)):
                if heap_size < K:
                    heap_size = heap_size + 1
                    heapq.heappush(score_heap, (scores[i], candidate_list[i]))
                else:
                    heapq.heappushpop(score_heap, (scores[i], candidate_list[i]))

            # print("DEBUG heap::",time.time()-s)
            for tm in self.timing_methods:
                time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
            scores, corpus_ids = list(zip(*score_heap))

        else:
            corpus_ids = candidate_list
            for tm in self.timing_methods:
                time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
        sort_timer_start = {}
        for tm in self.timing_methods:
            sort_timer_start[tm] = self.timing_funcs[tm]()
        # sort_zip_timer_start = time.time()
        # score_with_corpus_ids = list(zip(scores, corpus_ids))
        # time_dict['sort_zip_time'] = time.time() - sort_zip_timer_start
        # sorted_score_with_corpus_ids = sorted(score_with_corpus_ids, reverse=True)
        # sort_unzip_timer_start = time.time()
        # sorted_scores, sorted_corpus_ids = list(zip(*sorted_score_with_corpus_ids))
        # time_dict['sort_unzip_time'] = time.time() - sort_unzip_timer_start
        scores_arr = np.array(scores)
        corpus_ids_arr = np.array(corpus_ids)
        sorted_ids = np.argsort(-scores_arr)
        sorted_scores = scores_arr[sorted_ids]
        sorted_corpus_ids = corpus_ids_arr[sorted_ids]
        for tm in self.timing_methods:
            time_dict[tm]['sort_procedure_time'] = self.timing_funcs[tm]() - sort_timer_start[tm]
        other_data_dict['len_candidate_list'] = len(sorted_corpus_ids)
        return list(sorted_scores), list(sorted_corpus_ids), time_dict, other_data_dict

    def retrieve(self, q_embed, K, hash_mode=None, no_bucket=False, qid=None):
        """
            Input : query_embed : to compute actual asym dist
                      shape is (1*d)
            Input : K : top K similar items to return
            Output : top K items, time taken for retrieval, accuracy?

            given query and a number k, find the top k closest corpus items
            loop over al hash_tables:
              map query to corr bucket:
                compute Asymmetric similarity between query and each corpus item in bucket and update min heap
            return [TODO: decide metric] of corpus in min heap
            TODO: guard against too few corpus items?
        """
        # Given input query_embed: generate query_hashcode : to ID query bucket
        start_hashcode_gen = {}
        end_hashcode_gen = {}
        if not no_bucket:
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = self.timing_funcs[tm]()

            if self.av.HASH_MODE == "wmh":
                # WMH(DrHash) toolkit throws exception for zero vectors. Adding small random noise
                if all(q_embed[0] == 0):
                    q_embed = q_embed + np.random.normal(0, 1e-8, len(q_embed[0]))
                q_hashcode = get_wmh_hcode(self.av, q_embed)
            else:
                q_hashcode = self.preprocess_hashcodes(self.fetch_RH_hashcodes(q_embed, hash_mode, isQuery=True, qid=qid)).squeeze().cpu()

            for tm in self.timing_methods:
                end_hashcode_gen[tm] = self.timing_funcs[tm]()
        else:
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = 0
                end_hashcode_gen[tm] = 0

        start_candidate_list_gen = {}
        end_candidate_list_gen = {}
        if no_bucket:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]()
            # We consider all corpus items
            candidate_list = list(range(self.num_corpus_items))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()
        else:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]()
            # We use q hashcode to identify buckets, and take union of corpus items into candidate set
            candidate_list = []
            for table_id in range(self.num_hash_tables):
                # identify bucket
                bucket_id = self.assign_bucket(table_id, q_hashcode)
                candidate_list.extend(self.all_hash_tables[table_id][bucket_id])

            # remove duplicates from candidate_list
            candidate_list = list(set(candidate_list))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()

            if self.av.DEBUG:
                print("No. of candidates found", len(candidate_list))

        if hash_mode == "flora":
            scores, corpus_ids, time_dict, other_data_dict = self.heapify(q_embed, candidate_list, K, use_tensor=False, qid=qid)
        else:
            scores, corpus_ids, time_dict, other_data_dict = self.heapify(q_embed, candidate_list, K, use_tensor=False)
        for tm in self.timing_methods:
            time_dict[tm]['candidate_list_gen_time'] = end_candidate_list_gen[tm] - start_candidate_list_gen[tm]
            time_dict[tm]['hashcode_gen_time'] = end_hashcode_gen[tm] - start_hashcode_gen[tm]
        return len(candidate_list), scores, corpus_ids, time_dict, other_data_dict
