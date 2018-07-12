#!/usr/bin/env python
'''
| Filename    : nvbs.py
| Description : Neural Variational Bayesian Sets
| Author      : Pushpendre Rastogi
| Created     : Mon Sep 11 14:00:17 2017 (-0400)
| Last-Updated: Thu Sep 14 12:56:59 2017 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 125
'''
from concrete.search import SearchService
from concrete.search.ttypes import SearchCapability, SearchResult, SearchResultItem, SearchType
from concrete.uuid.ttypes import UUID
from concrete.entities.ttypes import Entity
from concrete.services.ttypes import ServiceInfo
from concrete.util import generate_UUID, read_communication_from_file
from concrete.util.file_io import CommunicationReader
from concrete.util.search_wrapper import SearchServiceWrapper

from nn_params import NN_PARAMS
from scipy.misc import logsumexp
import scipy, math
import scipy.sparse
import scipy.special
import scipy.misc
import numpy as np
import utils, codecs
import os, sys
from enum import Enum
saved_model = {}
LOG2PI = math.log(2) + math.log(math.pi)
opj = os.path.join
from os.path import exists as ope
import cPickle as pkl
import logging
logger = logging.getLogger(__name__)

class NVBSALGO(Enum):
    POSMEAN = 'Mean of posterior'
    KL = 'KL Divergence with query as truth'
    IKL = 'Inverse KL Divergence with item as truth'
    SYMKL = 'Average of KL and IKL'
    PDFIP = 'PDF Inner Product'
    PWMEAN = 'L2 Distance of Precision Weighted Mean'
    ELBOBS = 'ELBO based Bayesian Sets'


class NVBS(object):
    " Neural Variational Bayesian Sets -- BOW NVBS "
    version = '0.1'
    def __init__(self, data_set, nnp, method=NVBSALGO.PWMEAN, opts=None,
                 id2guid=None, guid2id=None, guid2name=None, guid2sent=None, id2feat=None):
        self.vocab_size = nnp.decoder_bias.shape[0]
        self.n_hidden = nnp.encoder_bias.shape[0]
        self.n_topic = nnp.mean_bias.shape[0]
        self.non_linearity = dict(tanh=np.tanh, sigmoid=scipy.special.expit,
                                  relu=lambda x: np.maximum(x, 0, x))[nnp.non_linearity]
        self.cX_csr, self.cX_csc = self.get_cX(self.vocab_size, data_set)
        self.X = len(data_set)
        self.decoder_mat = nnp.decoder_mat
        self.decoder_bias = nnp.decoder_bias
        self.nnp = nnp
        # ------------------------------------------------------ #
        # Compute {log p(x) | x in cX} via matrix multiplication #
        # ------------------------------------------------------ #
        self.enc_vec = self.non_linearity(self.cX_csc.dot(nnp.encoder_mat) + nnp.encoder_bias)
        self.mean = self.enc_vec.dot(nnp.mean_mat) + nnp.mean_bias
        self.logsigm = self.enc_vec.dot(nnp.sigma_mat) + nnp.sigma_bias
        self.sum_logsigm = self.logsigm.sum(axis=1)
        self.sigma = np.exp(self.logsigm)
        self.invsigma = np.reciprocal(self.sigma)
        self.method = method
        self.pwm = self.mean * self.invsigma
        self.id2guid = id2guid
        self.guid2id = guid2id
        self.guid2name = guid2name
        self.guid2sent = guid2sent
        self.feat2id = dict((b,a) for a,b in id2feat.iteritems())
        self.id2feat = id2feat
        assert len(id2feat) == self.decoder_bias.shape[0]
        if method == NVBSALGO.ELBOBS:
            self.elbo_n_sample = opts.elbo_n_sample
            self.gaussian_samples = np.random.randn(self.elbo_n_sample, self.n_topic)
        # self.elbo_px = np.zeros(self.X)
        # for i in xrange(self.X):
        #     if i%10 == 0:
        #         print( '%.4f'%(i / float(self.X)))
        #     self.elbo_px[i] = self.elbo([i], self.mean[i], self.sigma[i],
        #                                 self.logsigm, self.decoder_mat, self.decoder_bias)
        return

    @staticmethod
    def get_cX(vocab_size, data_set):
        I = []
        J = []
        D = []
        for doc_idx, doc in enumerate(data_set):
            for feat_idx, val in sorted(doc.items(), key=lambda x: x[1]):
                I.append(doc_idx)
                J.append(vocab_size - 1 if feat_idx == -1 else feat_idx)
                D.append(val)
        X = len(data_set)
        cX_coo = scipy.sparse.coo_matrix((D, (I, J)), shape=(X, vocab_size))
        cX_csr = cX_coo.tocsr()
        cX_csc = cX_coo.tocsc()
        return cX_csr, cX_csc

    # @profile
    def elbo(self, idi, mean, sigma, logsigma, decoder_mat, decoder_bias):
        d = mean.shape[0]
        arr = np.zeros(self.elbo_n_sample)
        for s in xrange(self.elbo_n_sample):
            e = self.gaussian_samples[s]
            z = e * np.sqrt(sigma) + mean
            log_qzx = -0.5  * ( (e * e).sum() + d * LOG2PI ) - logsigma.sum()
            activation = z.dot(decoder_mat) + decoder_bias
            lsm = activation - logsumexp(activation)
            log_pxz = self.cX_csr[idi].dot(lsm).sum()
            log_pz = -0.5 * ((z * z).sum() + d * LOG2PI)
            arr[s] = log_pz + log_pxz - log_qzx
        return logsumexp(arr)

    def score_elbobs(self, idi):
        sigma_inv, mean_sigma_inv = self.posterior_mean_sigma_intmdt(idi)
        sigma = np.zeros(self.n_topic)
        mean = np.zeros(self.n_topic)
        elbo_pxd = np.zeros(self.X)
        for i in xrange(self.X):
            if i%100 == 0:
                sys.stdout.write('%.4f\r'%(i / float(self.X)))
                sys.stdout.flush()
            # np.reciprocal(self.sigma[i], out=temp)
            sigma = np.reciprocal(sigma_inv + np.reciprocal(self.sigma[i]))
            mean = (mean_sigma_inv + (self.mean[i] * np.reciprocal(self.sigma[i]))) * sigma
            elbo_pxd[i] = self.elbo(idi + [i], mean, sigma, self.logsigm, self.decoder_mat, self.decoder_bias) - self.elbo([i], self.mean[i], self.sigma[i], self.logsigm, self.decoder_mat, self.decoder_bias) # self.elbo_px[i]
        return elbo_pxd


    def posterior_mean_sigma_intmdt(self, idi):
        # -------------------------------- #
        # Compute {log p(x, cD) | x in cX} #
        # -------------------------------- #
        # We just need to compute the new mean and sigma
        # The rest of the computation will remain the same.
        sigma_inv = np.zeros(self.n_topic)
        mean_sigma_inv = np.zeros(self.n_topic)
        # temp =  np.zeros(self.n_topic)
        for i in idi:
            temp = np.reciprocal(self.sigma[i])
            sigma_inv += temp
            mean_sigma_inv += self.mean[i] * temp
        return sigma_inv, mean_sigma_inv

    def weighted_posterior_mean_sigma_intmdt(self, idi, weights):
        sigma_inv = np.zeros(self.n_topic)
        mean_sigma_inv = np.zeros(self.n_topic)
        for enum_i, i in enumerate(idi):
            temp = np.reciprocal(self.sigma[i]) * abs(weights[enum_i])
            sigma_inv += temp
            if weights[enum_i] < 0:
                mean_sigma_inv -= self.mean[i] * temp
            else:
                mean_sigma_inv += self.mean[i] * temp
        return sigma_inv, mean_sigma_inv

    def posterior_mean_sigma(self, idi, weights):
        if weights is None:
            sigma_inv, mean_sigma_inv = self.posterior_mean_sigma_intmdt(idi)
        else:
            assert len(weights) == len(idi)
            sigma_inv, mean_sigma_inv = self.weighted_posterior_mean_sigma_intmdt(idi, weights)

        possigma = np.reciprocal(sigma_inv)
        posmean = mean_sigma_inv * possigma
        return posmean, possigma

    def score_posmean(self, posmean, possigma):
        return np.linalg.norm(self.mean - posmean, axis=1)

    def score_kl(self, posmean, possigma):
        S = self.invsigma
        mean = self.mean
        M = mean - posmean
        t1 = (S * possigma).sum(axis=1)
        t2 = (M*M*S).sum(axis=1)
        t3 = self.sum_logsigm - possigma.sum()
        return t1+t2+t3

    def score_ikl(self, posmean, possigma):
        S = np.reciprocal(possigma)
        M = self.mean - posmean
        t1 = (S * self.sigma).sum(axis=1)
        t2 = (M*M*S).sum(axis=1)
        t3 = self.sum_logsigm - possigma.sum()
        return t1+t2-t3

    def score_symkl(self, posmean, possigma):
        return self.score_kl(posmean, possigma) + self.score_ikl(posmean, possigma)

    def score_pdfip(self, posmean, possigma):
        S = self.sigma + possigma
        M = self.mean - posmean
        return np.log(S).sum(axis=1) + (M*M*np.reciprocal(S)).sum(axis=1)

    def score_pwmean(self, posmean, possigma):
        pwm = posmean * np.reciprocal(possigma)
        return np.linalg.norm(self.pwm - pwm, axis=1)

    def query(self, guids, n_results, weights):
        idi = [self.guid2id[g] for g in guids]
        posmean, possigma = self.posterior_mean_sigma(idi, weights)
        scores = posmean.dot(self.decoder_mat) + self.decoder_bias
        Z = scipy.misc.logsumexp(scores)
        log_distribution = scores - Z
        if self.method == NVBSALGO.POSMEAN:
            score = self.score_posmean(posmean, possigma)
        elif self.method == NVBSALGO.KL:
            score = self.score_kl(posmean, possigma)
        elif self.method == NVBSALGO.IKL:
            score = self.score_ikl(posmean, possigma)
        elif self.method == NVBSALGO.SYMKL:
            score = self.score_symkl(posmean, possigma)
        elif self.method == NVBSALGO.PDFIP:
            score = self.score_pdfip(posmean, possigma)
        elif self.method == NVBSALGO.PWMEAN:
            score = self.score_pwmean(posmean, possigma)
        elif self.method == NVBSALGO.ELBOBS:
            score = self.score_elbobs(posmean, possigma)
        else:
            raise ValueError(self.method)
        top_idi = score.argpartition(n_results)[:n_results]
        top_idi = sorted(top_idi, key=lambda i: score[i])
        entities = [(self.id2guid[i], -score[i]) for i in top_idi]
        return entities, log_distribution

    def query_similar_word(self, feat, emb=None):
        if emb is None:
            emb = self.nnp.encoder_mat[self.feat2id[feat]]
        hidden_layer = self.non_linearity(emb + self.nnp.encoder_bias)
        mean = hidden_layer.dot(self.nnp.mean_mat) + self.nnp.mean_bias
        logsigm = hidden_layer.dot(self.nnp.sigma_mat) + self.nnp.sigma_bias
        scores = mean.dot(self.decoder_mat) + self.decoder_bias
        # Z = scipy.misc.logsumexp()
        topfeat = [e[1] for e in sorted(self.id2feat.items(), key=lambda x: scores[x[0]], reverse=True)[:100]]
        print topfeat
        return

    def query_activated_words(self, i):
        scores = self.decoder_mat[i] + self.decoder_bias
        topfeat = [e[1] for e in sorted(self.id2feat.items(), key=lambda x: scores[x[0]], reverse=True)[:100]]
        print topfeat
        return

    def z_init(self, x, return_mean=False):
        hidden_layer = self.non_linearity(x.dot(self.nnp.encoder_mat) + self.nnp.encoder_bias)
        mean = hidden_layer.dot(self.nnp.mean_mat) + self.nnp.mean_bias
        logsigm = hidden_layer.dot(self.nnp.sigma_mat) + self.nnp.sigma_bias
        # TODO: Sample using mean and logsigm instead of returning mean.
        sigma_z = np.exp(logsigm).squeeze()
        if return_mean:
            return mean.squeeze(), sigma_z
        else:
            mvn_rv = np.random.randn(self.n_topic)
            return (sigma_z * mvn_rv + mean.squeeze()), sigma_z

    def hmc_step(self, z, grad_fn, ll_fn, eps_vec, inloop):
        '''z    : Initial value of z
        grad_fn : Fn that takes z' and gives gradient of log p(z,x) wrt z at z'.
        ll_fn   : Fn that takes z' and gives ll upto a additive value that does not depend on z.
        eps_vec : Vector of epsilon values.
        inloop  : (default 10)
        --- Return ---
        Sample and whether proposal was accepted or rejected.
        '''
        r = np.random.randn(self.n_topic)
        zp, rp = z.copy(), r.copy()
        for l in range(inloop):
            rp += 0.5 * eps_vec * grad_fn(zp)
            zp += eps_vec * rp
            rp += 0.5 * eps_vec * grad_fn(zp)
        thresh = (ll_fn(zp) - ll_fn(z)) - (rp.dot(rp) - r.dot(r))/2
        # thresh2 = (ll_fn(zp) - rp.dot(rp)/2) - (ll_fn(z) - r.dot(r)/2)
        # assert thresh == thresh2
        if (thresh >= 0 or np.log(1e-10  + np.random.rand()) < thresh):
            return zp, 1.0, thresh
        else:
            return z.copy(), 0.0, thresh

    def grad_fn_factory(self, x):
        def f(z):
            scores = z.dot(self.decoder_mat) + self.decoder_bias
            lse = scipy.misc.logsumexp(scores)
            scaled_dist = np.exp(scores - (lse - np.log(x.sum())))
            return self.decoder_mat.dot((x.toarray() - scaled_dist).squeeze()) - z.squeeze()
        return f

    def ll_fn_factory(self, x):
        def f(z):
            scores = z.dot(self.decoder_mat) + self.decoder_bias
            logits = scores - scipy.misc.logsumexp(scores)
            return x.dot(logits) - (z.dot(z) / 2)
        return f

    def hmc_sample(self, x, n_sample=1000, eps0=0.1, init_from_mean=False):
        '''
        This HMC sampling procedure and specially the setting of hyper-parameters such as
        - L: The size of internal loop
        - eps0: The value of global scaling of eps and target acceptance rate
        - Perturbing eps based on the value of sigma
        are based on
        "Learning Deep Latent Gaussian Models with Markov Chain Monte Carlo", Hoffman (2017)
        x        :
        n_sample : (default 1000)
        eps0     : (default 0.1)
        --- OUTPUT ---
        '''
        trace = []
        z, sigma_z = self.z_init(x, return_mean=init_from_mean)
        grad_fn = self.grad_fn_factory(x)
        ll_fn = self.ll_fn_factory(x)
        accept_ra = 1.0  # Acceptance running average.
        for i in range(n_sample):
            L = int(np.ceil(1.0 / eps0))  # 2. Choose L = 1 / eps0 so that we don't u-turn.
            eps_vec = eps0 * sigma_z      # 3. eps_vec[k] = eps0 * sigma[k]
            z, accept, thresh = self.hmc_step(z, grad_fn, ll_fn, eps_vec, L)
            trace.append(z)
            # Decrease eps0 if acceptance rate falls below 0.25
            eps0 *= (0.95 if accept_ra < 0.25 else 1.05)
            # Update the running average, slowly in the beginning, faster in the end.
            accept_ra += (accept - accept_ra) * (0.05 if i < 100 else 0.5)
            print 'thresh', thresh, 'eps0', eps0, 'accept', accept, 'accept_ra', accept_ra
        return trace  # Do burning, thinning afterwards.


def lm_score(lp, feat2id, tokens):
    pure_score = np.mean([lp[feat2id[e]] for e in tokens if e in feat2id])
    penalty = -100 if len(tokens) < 5 else 0
    return pure_score + penalty

class EntitySearchProvider(SearchService.Iface):
    def __init__(self, lang, index, k_query, k_rationale):
        self.lang = lang
        self.index = index
        self.name = index.__class__.__name__
        self.version = index.version
        self.k_query = k_query
        self.k_rationale = k_rationale
        return

    def about(self):
        logger.info("Received about() call")
        service_info = ServiceInfo()
        service_info.name = self.name
        service_info.version = self.version
        return service_info

    def alive(self):
        logger.info("Received alive() call")
        return True

    def getCapabilities(self):
        logger.info("Received getCapabilities() call")
        search_capabilities = []

        communications_capability = SearchCapability()
        communications_capability.lang = self.lang
        communications_capability.type = SearchType.ENTITIES
        search_capabilities.append(communications_capability)

        return search_capabilities

    def getCorpora(self):
        logger.info("Received getCorpora() call")
        return []


    def search(self, query):
        logger.info("Received SearchQuery: '%s'" % query)
        search_result_items = []
        weights = (None
                   if query.labels is None or len(query.labels) == 0
                   else [float(e) for e in query.labels])
        entities, log_distribution = self.index.query(query.terms, query.k, weights=weights)
        lm = sorted(self.index.feat2id.items(),
                    key=lambda x: log_distribution[self.index.feat2id[x[0]]], reverse=True)
        query.labels = [e[0] for e in lm[:self.k_query]]
        for guid, score in entities:
            search_result_item = SearchResultItem()
            uuid_sentences = self.index.guid2sent[guid]
            ss = np.empty((len(uuid_sentences),))
            for idx, (uuid, sent) in enumerate(uuid_sentences):
                ss[idx] = lm_score(log_distribution, self.index.feat2id, sent)
            sorted_idi = np.argsort(ss)[-1:-self.k_rationale-1:-1]
            sents = [' '.join(uuid_sentences[e][1]) for e in sorted_idi]
            uuidi = [uuid_sentences[e][0] for e in sorted_idi]
            search_result_item.communicationId = guid+'\n'+'\n'.join(sents)
            search_result_item.sentenceId = None
            search_result_item.score = score
            entity = Entity()
            entity.uuid = generate_UUID()
            entity.id = guid
            uuidList = []
            for single_uuid in uuidi:
                uuidObj = UUID()
                uuidObj.uuidString = single_uuid
                uuidList.append(uuidObj)
            entity.mentionIdList = uuidList
            search_result_item.entity = entity
            search_result_items.append(search_result_item)

        search_result = SearchResult()
        search_result.uuid = generate_UUID()
        search_result.searchResultItems = search_result_items
        search_result.searchQuery = query
        logger.info("Returned SearchResult with %d SearchResultItems\n" % len(search_result.searchResultItems))
        return search_result



def serve():
    train_test_url = opj(args.data_dir, 'train_test.feat')
    entity_map_url = opj(args.data_dir, 'entity.map')
    feat_map_url = opj(args.data_dir, 'vocab.new')
    entity_sent_url = opj(args.data_dir, 'entities.sentences')
    guid2name = {}
    guid2id = {}
    id2guid = {}
    guid2sent = {}
    # The train_test.feat file contains some entities such as number 1997
    # that has no features. Its feature line is blank.
    # These entities were removed while training the neural network architecture.
    # Therefore to map the embeddings in NVGE back to the KB we need to use this
    # alignment information. This information is not necessary for BS because BS
    # can easily handle the fact that some entities have no features (ie. the )
    # document is empty.
    data_set, data_count, alignment = utils.data_set(train_test_url)
    for idx, row in enumerate(codecs.open(entity_map_url, 'r', 'utf-8').read().split('\n')):
        if row == '': continue
        dbid, canonical = row.split('\t')
        guid2name[dbid] = canonical
        if idx in alignment:
            guid2id[dbid] = alignment[idx]
            id2guid[alignment[idx]] = dbid

    GUID2SENT_PKL_FILE = opj(args.data_dir, os.path.pardir, 'guid2sent.pkl')
    try:
        print 'Loading', GUID2SENT_PKL_FILE
        guid2sent = pkl.load(open(GUID2SENT_PKL_FILE))
    except:
        print 'Could not find', GUID2SENT_PKL_FILE
        concrete_entity_files = os.listdir(args.concrete_entity_dir)
        for commidx, filename in enumerate(concrete_entity_files):
            print '%-5d\r'%((commidx * 100) / len(concrete_entity_files)),
            comm = read_communication_from_file(opj(args.concrete_entity_dir, filename))
            guid = comm.id
            for sent in comm.sectionList[0].sentenceList:
                uuid = sent.uuid.uuidString
                tokens = [e.text for e in sent.tokenization.tokenList.tokenList]
                try:
                    guid2sent[guid].append((uuid, tokens))
                except KeyError:
                    guid2sent[guid] = [(uuid, tokens)]
        with open(GUID2SENT_PKL_FILE, 'wb') as gpf:
            print 'Dumping', GUID2SENT_PKL_FILE
            pkl.dump(guid2sent, gpf)

    # for row in codecs.open(entity_sent_url, 'r', 'utf-8').read().split('\n'):
    #     row = row.split(' ||| ')
    #     guid = row[0]
    #     for sent in row[1:]:
    #         tokens = sent.split()
    #         try:
    #             guid2sent[guid].append(tokens)
    #         except KeyError:
    #             guid2sent[guid] = [tokens]
    id2feat_data = codecs.open(feat_map_url, 'r', 'utf-8').read().split('\n')
    id2feat = dict((((sum(1 for e in id2feat_data if e != '') -1) if idx == 0 else (idx-1)), row.split()[0])
                   for idx, row in enumerate(id2feat_data)
                   if row != '')
    print('Checking feature size =', len(data_set[guid2id[":Entity_ENG_EDL_0092354"]]),
          'for', guid2name[":Entity_ENG_EDL_0092354"],
          'max(id2feat.values())', max(id2feat.keys()))
    def load(args):
        import cPickle as pkl
        with open(opj(args.data_dir, args.model_pkl), 'rb') as f:
            nnp = pkl.load(f)
        return nnp

    handler = EntitySearchProvider(args.language,
                                   NVBS(data_set=data_set, nnp=load(args),
                                        method=getattr(NVBSALGO, args.algorithm),
                                        opts=args,
                                        id2guid=id2guid,
                                        guid2id=guid2id,
                                        guid2name=guid2name,
                                        guid2sent=guid2sent,
                                        id2feat=id2feat),
                                   args.k_query,
                                   args.k_rationale)
    server = SearchServiceWrapper(handler)
    if args.serve:
        print('Starting NVBS Server')
        server.serve(args.host, args.port)
    else:
        return handler.index


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Neural Variational Recommender System')
    arg_parser.add_argument('--data_dir', default='data/tac2017', type=str)
    arg_parser.add_argument('--model_pkl', default='tac2017.nvdm.pkl', type=str)
    arg_parser.add_argument('--elbo_n_sample', default=1, type=int, help='Number of samples.')
    arg_parser.add_argument('--algorithm', default='PWMEAN', type=str)
    arg_parser.add_argument('--port', default=12360, type=int, help='Port of Bayesian Sets server')
    arg_parser.add_argument('--host', default='localhost', type=str)
    arg_parser.add_argument('--n_results', type=int, default=10, help='Number of results returned by BS server')
    arg_parser.add_argument('--language', help='ISO 639-2/T language code', default='eng', type=str)
    arg_parser.add_argument('--k_query', default=10, type=int)
    arg_parser.add_argument('--k_rationale', default=10, type=int)
    arg_parser.add_argument('--concrete_entity_dir', default='data_dir/tac2017.concrete.entities', type=str)
    arg_parser.add_argument('--serve', default=1, type=int)
    args=arg_parser.parse_args()
    # args.concrete_entity_dir = opj(args.data_dir, args.concrete_entity_dir)
    assert ope(args.data_dir), args.data_dir
    assert ope(opj(args.data_dir, args.model_pkl)), opj(args.data_dir, args.model_pkl)
    assert ope(args.concrete_entity_dir), args.concrete_entity_dir
    print 'ope(opj(args.data_dir, os.path.pardir, \'guid2sent.pkl\'))', ope(opj(args.data_dir, os.path.pardir, 'guid2sent.pkl'))
    # import random
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    server = serve()
